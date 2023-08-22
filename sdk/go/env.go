package dagger

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	goast "go/ast"
	"go/parser"
	"go/token"
	"net"
	"net/http"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"

	"dagger.io/dagger/querybuilder"
	"github.com/Khan/genqlient/graphql"
	"github.com/iancoleman/strcase"
	"github.com/vektah/gqlparser/v2/ast"
)

var (
	errorT      = reflect.TypeOf((*error)(nil)).Elem()
	marshallerT = reflect.TypeOf((*querybuilder.GraphQLMarshaller)(nil)).Elem()
	contextT    = reflect.TypeOf((*context.Context)(nil)).Elem()
)

var resolvers = map[string]*goFunc{}

var getEnv bool

func getClientParams() (graphql.Client, *querybuilder.Selection) {
	portStr, ok := os.LookupEnv("DAGGER_SESSION_PORT")
	if !ok {
		panic("DAGGER_SESSION_PORT is not set")
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		panic(fmt.Errorf("DAGGER_SESSION_PORT %q is invalid: %w", portStr, err))
	}

	sessionToken := os.Getenv("DAGGER_SESSION_TOKEN")
	if sessionToken == "" {
		panic("DAGGER_SESSION_TOKEN is not set")
	}

	host := fmt.Sprintf("127.0.0.1:%d", port)

	dialTransport := &http.Transport{
		DialContext: func(_ context.Context, _, _ string) (net.Conn, error) {
			return net.Dial("tcp", host)
		},
	}
	httpClient := &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			r.SetBasicAuth(sessionToken, "")
			return dialTransport.RoundTrip(r)
		}),
	}
	gqlClient := graphql.NewClient(fmt.Sprintf("http://%s/query", host), httpClient)

	return gqlClient, querybuilder.Query()
}

func Serve(r *Environment) {
	ctx := context.Background()
	if getEnv {
		id, err := r.ID(ctx)
		if err != nil {
			writeErrorf(err)
		}
		err = os.WriteFile("/env/id", []byte(id), 0644)
		if err != nil {
			writeErrorf(err)
		}
		return
	}

	inputBytes, err := os.ReadFile("/inputs/dagger.json")
	if err != nil {
		writeErrorf(fmt.Errorf("unable to open request file: %w", err))
	}
	var input struct {
		Resolver string
		Parent   map[string]any
		Args     map[string]any
	}
	if err := json.Unmarshal(inputBytes, &input); err != nil {
		writeErrorf(fmt.Errorf("unable to parse request file: %w", err))
	}

	if input.Resolver == "" {
		writeErrorf(fmt.Errorf("missing resolver"))
	}
	fieldName := input.Resolver
	fn := resolvers[fieldName]

	var result any
	if fn == nil {
		// trivial resolver
		if input.Parent != nil {
			result = input.Parent[fieldName]
		}
	} else {
		result, err = fn.call(ctx, input.Parent, input.Args)
		if err != nil {
			writeErrorf(err)
		}
	}
	if result == nil {
		result = make(map[string]any)
	}

	if fn.resultConverter == nil {
		fn.resultConverter = convertResult
	}
	result, err = fn.resultConverter(ctx, result)
	if err != nil {
		writeErrorf(err)
	}
	output, err := json.Marshal(result)
	if err != nil {
		writeErrorf(fmt.Errorf("unable to marshal response: %v", err))
	}
	if err := os.WriteFile("/outputs/dagger.json", output, 0600); err != nil {
		writeErrorf(fmt.Errorf("unable to write response file: %v", err))
	}
}

func WithCheck(r *Environment, in any) *Environment {
	if _, ok := in.(*Check); ok {
		// just let the codegen sdk caller handle this
		return nil
	}
	flag.Parse()

	// TODO: dedupe huge chunks of code
	typ := reflect.TypeOf(in)
	if typ.Kind() != reflect.Func {
		writeErrorf(fmt.Errorf("expected func, got %v", typ))
	}
	val := reflect.ValueOf(in)
	name := runtime.FuncForPC(val.Pointer()).Name()
	if name == "" {
		writeErrorf(fmt.Errorf("anonymous functions are not supported"))
	}
	fn := &goFunc{
		name: name,
		typ:  typ,
		val:  val,
	}
	for i := 0; i < fn.typ.NumIn(); i++ {
		inputParam := fn.typ.In(i)
		fn.args = append(fn.args, &goParam{
			typ: inputParam,
		})
	}
	for i := 0; i < fn.typ.NumOut(); i++ {
		outputParam := fn.typ.Out(i)
		fn.returns = append(fn.returns, &goParam{
			typ: outputParam,
		})
	}
	if len(fn.returns) > 2 {
		writeErrorf(fmt.Errorf("expected 1 or 2 return values, got %d", len(fn.returns)))
	}

	filePath, lineNum := fn.srcPathAndLine()
	// TODO: cache parsed files
	fileSet := token.NewFileSet()
	parsed, err := parser.ParseFile(fileSet, filePath, nil, parser.ParseComments)
	if err != nil {
		writeErrorf(fmt.Errorf("parse file: %w", err))
	}
	goast.Inspect(parsed, func(n goast.Node) bool {
		if n == nil {
			return false
		}
		switch decl := n.(type) {
		case *goast.FuncDecl:
			astStart := fileSet.PositionFor(decl.Pos(), false)
			astEnd := fileSet.PositionFor(decl.End(), false)
			// lineNum can be inside the function body due to optimizations that set it to
			// the location of the return statement
			if lineNum < astStart.Line || lineNum > astEnd.Line {
				return true
			}

			fn.name = decl.Name.Name
			fn.doc = strings.TrimSpace(decl.Doc.Text())

			fnArgs := fn.args
			if decl.Recv != nil {
				// ignore the object receiver for now
				fnArgs = fnArgs[1:]
				fn.hasReceiver = true
			}
			astParamList := decl.Type.Params.List
			argIndex := 0
			for _, param := range astParamList {
				// if the signature is like func(a, b string), then a and b are in the same Names slice
				for _, name := range param.Names {
					fnArgs[argIndex].name = name.Name
					argIndex++
				}
			}
			return false

		case *goast.GenDecl:
		default:
		}
		return true
	})

	check := dag.Check()
	resolvers[lowerCamelCase(fn.name)] = fn

	if !getEnv {
		return r
	}

	check = check.
		WithName(strcase.ToLowerCamel(fn.name)).
		WithDescription(fn.doc)

	return r.WithCheck(check)
}

type goTypes struct {
	Structs map[string]*goStruct
	Funcs   map[string]*goFunc
}

type goStruct struct {
	name    string
	typ     reflect.Type
	fields  []*goField
	methods []*goFunc
	doc     string
	// should this become a field of the Query type?
	topLevel bool
	// should this become a graphql Input type instead of an Object type?
	usedAsInput bool
}

type goField struct {
	name string
	typ  reflect.Type
}

type goFunc struct {
	name string
	// args are args to the function, except for the receiver
	// (if it's a method) and for the dagger.Context arg.
	args    []*goParam
	returns []*goParam
	typ     reflect.Type
	val     reflect.Value
	// TODO:
	// receiver *goStruct // only set for methods
	hasReceiver     bool
	doc             string
	resultConverter func(context.Context, any) (any, error)
}

type goParam struct {
	name     string
	doc      string
	typ      reflect.Type
	optional bool
}

func (fn *goFunc) srcPathAndLine() (string, int) {
	pc := fn.val.Pointer()
	fun := runtime.FuncForPC(pc)
	return fun.FileLine(pc)
}

func (fn *goFunc) call(ctx context.Context, rawParent, rawArgs map[string]any) (any, error) {
	var callArgs []reflect.Value
	if fn.hasReceiver {
		callArgs = append(callArgs, reflect.New(fn.args[0].typ).Elem())
	}

	for _, arg := range fn.args[len(callArgs):] {
		if arg.typ.Implements(contextT) {
			callArgs = append(callArgs, reflect.ValueOf(ctx))
			continue
		}

		rawArg, argValuePresent := rawArgs[arg.name]

		// support FooOpts structs
		if arg.typ.Kind() == reflect.Struct {
			opts := reflect.New(arg.typ)
			for i := 0; i < arg.typ.NumField(); i++ {
				field := arg.typ.Field(i)
				rawArg, optPresent := rawArgs[strcase.ToLowerCamel(field.Name)]
				if optPresent {
					optVal, err := convertInput(rawArg, field.Type)
					if err != nil {
						return nil, fmt.Errorf("unable to convert arg %s: %w", arg.name, err)
					}
					opts.Elem().Field(i).Set(reflect.ValueOf(optVal))
				}
			}
			callArgs = append(callArgs, opts.Elem())
			continue
		}

		switch {
		case argValuePresent:
			argVal, err := convertInput(rawArg, arg.typ)
			if err != nil {
				return nil, fmt.Errorf("unable to convert arg %s: %w", arg.name, err)
			}
			callArgs = append(callArgs, reflect.ValueOf(argVal))
		case !arg.optional:
			return nil, fmt.Errorf("missing required argument %s", arg.name)
		default:
			callArgs = append(callArgs, reflect.New(arg.typ).Elem())
		}
	}

	reflectOutputs := fn.val.Call(callArgs)
	var returnVal any
	var returnErr error
	for _, output := range reflectOutputs {
		if output.Type() == errorT {
			if !output.IsNil() {
				returnErr = output.Interface().(error)
			}
		} else {
			returnVal = output.Interface()
		}
	}
	return returnVal, returnErr
}

func init() {
	// TODO:(sipsma) silly+unmaintainable, is there a pre-made list of these somewhere?
	// Or can we have a rule that "ALLCAPS" becomes "allcaps" instead of aLLCAPS?
	strcase.ConfigureAcronym("URL", "url")
	strcase.ConfigureAcronym("CI", "ci")
	strcase.ConfigureAcronym("SDK", "sdk")
}

func lowerCamelCase(s string) string {
	return strcase.ToLowerCamel(s)
}

func inputName(name string) string {
	return name + "Input"
}

func goReflectTypeToGraphqlType(t reflect.Type, isInput bool) (*ast.Type, error) {
	// Handle types that implement the GraphQL serializer
	if t.Implements(marshallerT) {
		typ := reflect.New(t.Elem())
		var typeName string
		if isInput {
			result := typ.MethodByName(querybuilder.GraphQLMarshallerIDType).Call([]reflect.Value{})
			typeName = result[0].String()
		} else {
			result := typ.MethodByName(querybuilder.GraphQLMarshallerType).Call([]reflect.Value{})
			typeName = result[0].String()
		}
		return ast.NonNullNamedType(typeName, nil), nil
	}

	switch t.Kind() {
	case reflect.String:
		// /* TODO:(sipsma)
		// Huge hack: handle any scalar type from the go SDK (i.e. DirectoryID/ContainerID)
		// The much cleaner approach will come when we integrate this code w/ the
		// in-progress codegen work.
		// */
		// if strings.HasPrefix(t.PkgPath(), "dagger.io/dagger") {
		// 	return ast.NonNullNamedType(t.Name(), nil), nil
		// }
		return ast.NonNullNamedType("String", nil), nil
	case reflect.Int:
		return ast.NonNullNamedType("Int", nil), nil
	case reflect.Float32, reflect.Float64:
		// TODO:(sipsma) does this actually handle both float32 and float64?
		return ast.NonNullNamedType("Float", nil), nil
	case reflect.Bool:
		return ast.NonNullNamedType("Boolean", nil), nil
	case reflect.Slice:
		elementType, err := goReflectTypeToGraphqlType(t.Elem(), isInput)
		if err != nil {
			return nil, err
		}
		return ast.NonNullListType(elementType, nil), nil
	case reflect.Struct:
		if isInput {
			return ast.NonNullNamedType(inputName(t.Name()), nil), nil
		}
		return ast.NonNullNamedType(t.Name(), nil), nil // TODO:(sipsma) doesn't handle anything from another package (besides the sdk)
	case reflect.Pointer:
		nonNullType, err := goReflectTypeToGraphqlType(t.Elem(), isInput)
		if err != nil {
			return nil, err
		}
		nonNullType.NonNull = false
		return nonNullType, nil
	default:
		// TODO: return nil, fmt.Errorf("unsupported type %s", t.Kind())
		return nil, fmt.Errorf("unsupported type %s %s %s", t.Kind(), t.Name(), string(debug.Stack()))
	}
}

// convertResult will recursively walk the result and update any values that can
// be converted into a graphql ID to that. It also fixes the casing of any fields
// to match the casing of the graphql schema (lower camel case).
// TODO: the MarshalGQL func in querybuilder is very similar to this one, dedupe somehow?
func convertResult(ctx context.Context, result any) (any, error) {
	if result == nil {
		return result, nil
	}

	if result, ok := result.(querybuilder.GraphQLMarshaller); ok {
		id, err := result.XXX_GraphQLID(ctx)
		if err != nil {
			return nil, err
		}
		// ID-able dagger objects are serialized as their ID string across the wire
		// between the session and project container.
		return id, nil
	}

	switch typ := reflect.TypeOf(result).Kind(); typ {
	case reflect.Pointer:
		return convertResult(ctx, reflect.ValueOf(result).Elem().Interface())
	case reflect.Interface:
		return convertResult(ctx, reflect.ValueOf(result).Elem().Interface())
	case reflect.Slice:
		slice := reflect.ValueOf(result)
		for i := 0; i < slice.Len(); i++ {
			converted, err := convertResult(ctx, slice.Index(i).Interface())
			if err != nil {
				return nil, err
			}
			slice.Index(i).Set(reflect.ValueOf(converted))
		}
		return slice.Interface(), nil
	case reflect.Struct:
		converted := map[string]any{}
		for i := 0; i < reflect.TypeOf(result).NumField(); i++ {
			field := reflect.TypeOf(result).Field(i)
			value := reflect.ValueOf(result).Field(i).Interface()
			convertedField, err := convertResult(ctx, value)
			if err != nil {
				return nil, err
			}
			converted[lowerCamelCase(field.Name)] = convertedField
		}
		return converted, nil
	case reflect.Map:
		converted := map[string]any{}
		for _, key := range reflect.ValueOf(result).MapKeys() {
			value := reflect.ValueOf(result).MapIndex(key).Interface()
			convertedValue, err := convertResult(ctx, value)
			if err != nil {
				return nil, err
			}
			converted[lowerCamelCase(key.String())] = convertedValue
		}
		return converted, nil
	default:
		return result, nil
	}
}

// TODO: doc, basically inverse of convertResult
func convertInput(input any, desiredType reflect.Type) (any, error) {
	// check if desiredType implements querybuilder.GraphQLMarshaller, in which case it's a core type e.g. Container
	if desiredType.Implements(marshallerT) {
		// ID-able dagger objects are serialized as their ID string across the wire
		// between the session and project container.
		id, ok := input.(string)
		if !ok {
			return nil, fmt.Errorf("expected id to be a string, got %T(%+v)", input, input)
		}
		if desiredType.Kind() != reflect.Ptr {
			// just assuming it's always a pointer for now, not actually important
			return nil, fmt.Errorf("expected desiredType to be a pointer, got %s", desiredType.Kind())
		}
		desiredType = desiredType.Elem()

		// TODO: Add a .XXX_GraphQLObject(id) method to the generated SDK code to make this simpler + more maintainable
		graphqlType := reflect.New(desiredType).Interface().(querybuilder.GraphQLMarshaller).XXX_GraphQLType()
		switch graphqlType {
		case "Container":
			return dag.Container(ContainerOpts{
				ID: ContainerID(id),
			}), nil
		case "Directory":
			return dag.Directory(DirectoryOpts{
				ID: DirectoryID(id),
			}), nil
		case "Socket":
			return dag.Socket(SocketOpts{
				ID: SocketID(id),
			}), nil
		case "File":
			return dag.File(FileID(id)), nil
		case "Secret":
			return dag.Secret(SecretID(id)), nil
		case "Cache":
			cacheID := CacheID(id)
			return CacheVolume{
				q:  dag.q,
				c:  dag.c,
				id: &cacheID,
			}, nil
		default:
			return nil, fmt.Errorf("unhandled GraphQL marshaller type %s", graphqlType)
		}
	}

	// recurse
	inputObj := reflect.ValueOf(input)
	switch desiredType.Kind() {
	case reflect.Pointer:
		x, err := convertInput(inputObj.Interface(), desiredType.Elem())
		if err != nil {
			return nil, err
		}
		ptr := reflect.New(desiredType.Elem())
		ptr.Elem().Set(reflect.ValueOf(x))
		return ptr.Interface(), nil
	case reflect.Slice:
		returnObj := reflect.MakeSlice(desiredType, inputObj.Len(), inputObj.Len())
		for i := 0; i < inputObj.Len(); i++ {
			value := inputObj.Index(i).Interface()
			convertedValue, err := convertInput(value, desiredType.Elem())
			if err != nil {
				return nil, err
			}
			returnObj.Index(i).Set(reflect.ValueOf(convertedValue))
		}
		return returnObj.Interface(), nil
	case reflect.Struct:
		returnObj := reflect.New(desiredType).Elem()
		for i := 0; i < desiredType.NumField(); i++ {
			inputMap, ok := input.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("expected input to be a map[string]any, got %T", input)
			}
			value := inputMap[lowerCamelCase(desiredType.Field(i).Name)]
			desiredValueType := desiredType.Field(i).Type
			convertedField, err := convertInput(value, desiredValueType)
			if err != nil {
				return nil, err
			}
			returnObj.Field(i).Set(reflect.ValueOf(convertedField))
		}
		return returnObj.Interface(), nil
	case reflect.Map:
		returnObj := reflect.MakeMap(desiredType)
		for _, key := range inputObj.MapKeys() {
			value := inputObj.MapIndex(key).Interface()
			convertedValue, err := convertInput(value, desiredType.Elem())
			if err != nil {
				return nil, err
			}
			inputObj.SetMapIndex(key, reflect.ValueOf(convertedValue))
		}
		return returnObj.Interface(), nil
	default:
		return input, nil
	}
}

func writeErrorf(err error) {
	fmt.Println(err.Error())
	os.Exit(1)
}

// TODO: pollutes namespace, move to non internal package in dagger.io/dagger
type roundTripperFunc func(*http.Request) (*http.Response, error)

func (fn roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}
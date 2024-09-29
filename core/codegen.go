package core

import (
	"github.com/vektah/gqlparser/v2/ast"
)

type Codegen struct {
	Query *Query
	deps  *ModDeps
}

func (*Codegen) Type() *ast.Type {
	return &ast.Type{
		NamedType: "Codegen",
		NonNull:   true,
	}
}

func (*Codegen) TypeDescription() string {
	return "Configuration for generating SDK code based on API schema"
}

// TODO
func (*Codegen) Introspect() {
}

// TODO
func (*Codegen) GenerateClient() {
}

// TODO
func (*Codegen) GenerateModule() {
}

func newSyntax(t []*TypeDef, c LanguageConventions) *ClientBindings {
	s := &ClientBindings{
		typeDefs: t,
		Config: ClientBindingsConfig{
			LanguageConventions: c,
		},
	}
	for _, typeDef := range s.typeDefs {
		switch typeDef.Kind {
		case TypeDefKindScalar:
			s.Scalars = append(s.Scalars, newScalarType(typeDef.AsScalar.Value, s))
		case TypeDefKindEnum:
			s.Enums = append(s.Enums, newEnumType(typeDef.AsEnum.Value, s))
		case TypeDefKindObject:
			s.Objects = append(s.Objects, newObjectType(typeDef.AsObject.Value, s))
		}
	}
	return s
}

// ClientBindings is an abstraction for generating language specific client bindings.
type ClientBindings struct {
	Config     ClientBindingsConfig
	Scalars    []*ScalarType
	Enums      []*EnumType
	Objects    []*ObjectType
	Interfaces []*IntefaceType
	typeDefs   []*TypeDef
}

type ClientBindingsConfig struct {
	LanguageConventions LanguageConventions
}

type LanguageConventions struct {
	Scalar      NamingConvention
	Enum        NamingConvention
	EnumValue   NamingConvention
	Input       NamingConvention
	InputField  NamingConvention
	Object      NamingConvention
	Interface   NamingConvention
	Function    NamingConvention
	FunctionArg NamingConvention
}

type NamingConvention string

const (
	PascalCase         NamingConvention = "PASCAL_CASE"
	CamelCase          NamingConvention = "CAMEL_CASE"
	KebabCase          NamingConvention = "KEBAB_CASE"
	SnakeCase          NamingConvention = "SNAKE_CASE"
	ScreamingSnakeCase NamingConvention = "SCREAMING_SNAKE_CASE"
)

func newScalarType(t *ScalarTypeDef, c *ClientBindings) *ScalarType {
	return &ScalarType{
		Name:         newName(t.Name, c.Config.LanguageConventions.Scalar),
		Description:  t.Description,
		SourceModule: t.SourceModuleName,
	}
}

type ScalarType struct {
	Name         *Name
	Description  string
	SourceModule string
}

func newEnumType(t *EnumTypeDef, c *ClientBindings) *EnumType {
	values := make([]*EnumValue, 0, len(t.Values))
	for _, v := range t.Values {
		values = append(values, &EnumValue{
			Name:        newName(v.Name, NamingConvention("")),
			Description: v.Description,
		})
	}
	return &EnumType{
		Name:         newName(t.Name, c.Config.LanguageConventions.Enum),
		Description:  t.Description,
		Values:       values,
		SourceModule: t.SourceModuleName,
	}
}

type EnumType struct {
	Name         *Name
	Description  string
	Values       []*EnumValue
	SourceModule string
}

type EnumValue struct {
	Name        *Name
	Description string
}

type TypeRef struct {
	Kind       string // TODO: TypeKind
	IsNullable bool
}

type InputObjectType struct {
	Name        *Name
	Description string
	Fields      []*InputField
}

type InputField struct {
	Name              *Name
	IsNullable        bool
	IsDeprecated      bool
	DeprecationReason string
}

func newObjectType(t *ObjectTypeDef, c *ClientBindings) *ObjectType {
	funcs := make([]*ObjectFunction, 0, len(t.Fields)+len(t.Functions))
	for _, f := range t.Fields {
		funcs = append(funcs, newObjectFieldFunction(f, c))
	}
	for _, f := range t.Functions {
		funcs = append(funcs, newObjectFunction(f, c))
	}
	return &ObjectType{
		Name:        newName(t.Name, c.Config.LanguageConventions.Object),
		Description: t.Description,
		Functions:   funcs,
	}
}

type ObjectType struct {
	Name         *Name
	Description  string
	SourceModule string
	Functions    []*ObjectFunction
}

type IntefaceType struct {
	Name         string
	NameWords    []string
	SourceModule string
	Functions    []*ObjectFunction
}

func newObjectFieldFunction(t *FieldTypeDef, c *ClientBindings) *ObjectFunction {
	return &ObjectFunction{
		Name:        newName(t.Name, c.Config.LanguageConventions.Function),
		Description: t.Description,
		isField:     true,
	}
}

func newObjectFunction(t *Function, c *ClientBindings) *ObjectFunction {
	args := make([]*InputValue, 0, len(t.Args))
	for _, a := range t.Args {
		args = append(args, newInputValue(a, c))
	}
	return &ObjectFunction{
		Name:        newName(t.Name, c.Config.LanguageConventions.Function),
		Description: t.Description,
		HasArgs:     len(args) > 0,
		Args:        args,
	}
}

type ObjectFunction struct {
	Name            *Name
	Description     string
	HasArgs         bool
	HasOptionalArgs bool
	HasRequiredArgs bool
	Args            []*InputValue
	isField         bool
}

func newInputValue(t *FunctionArg, c *ClientBindings) *InputValue {
	return &InputValue{
		Name:         newName(t.Name, c.Config.LanguageConventions.FunctionArg),
		Description:  t.Description,
		DefaultValue: t.DefaultValue,
		IsNullable:   t.TypeDef.Optional,
		IsOptional:   t.TypeDef.Optional || t.DefaultValue == nil,
	}
}

type InputValue struct {
	Name         *Name
	Description  string
	DefaultValue JSON
	IsNullable   bool
	IsOptional   bool
}

func newName(original string, conv NamingConvention) *Name {
	words := []string{}
	n := &Name{
		Original: original,
		Words:    words,
	}
	switch conv {
	case PascalCase:
		n.Formatted = n.AsPascal()
	case CamelCase:
		n.Formatted = n.AsCamel()
	case KebabCase:
		n.Formatted = n.AsKebab()
	case SnakeCase:
		n.Formatted = n.AsSnake()
	default:
		n.Formatted = original
	}
	return n
}

type Name struct {
	Original  string
	Formatted string
	Words     []string
}

func (n *Name) AsPascal() string {
	return ""
}

func (n *Name) AsCamel() string {
	return ""
}

func (n *Name) AsKebab() string {
	return ""
}

func (n *Name) AsSnake() string {
	return ""
}

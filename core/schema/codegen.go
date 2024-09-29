package schema

import (
	"context"
	"fmt"

	"github.com/dagger/dagger/core"
	"github.com/dagger/dagger/dagql"
	"github.com/moby/buildkit/identity"
)

type codegenSchema struct {
	srv *dagql.Server
}

var _ SchemaResolvers = &codegenSchema{}

func (s *codegenSchema) Install() {
	dagql.Fields[*core.Query]{
		dagql.Func("codegen", s.codegen).
			Doc("Configuration for generating SDK code based on API schema"),
	}.Install(s.srv)

	dagql.Fields[*core.Engine]{
		dagql.Func("localCache", s.localCache).
			Doc("The local (on-disk) cache for the Dagger engine"),
	}.Install(s.srv)

	dagql.Fields[*core.EngineCache]{
		dagql.Func("entrySet", s.cacheEntrySet).
			Impure("Cache is changing asynchronously in the background").
			Doc("The current set of entries in the cache"),
		dagql.Func("prune", s.cachePrune).
			Impure("Mutates mutable state").
			Doc("Prune the cache of releaseable entries"),
	}.Install(s.srv)

	dagql.Fields[*core.EngineCacheEntrySet]{
		dagql.Func("entries", s.cacheEntrySetEntries).
			Doc("The list of individual cache entries in the set"),
	}.Install(s.srv)

	dagql.Fields[*core.EngineCacheEntry]{}.Install(s.srv)
}

func (s *codegenSchema) codegen(ctx context.Context, parent *core.Query, args struct{}) (*core.Codegen, error) {
	return &core.Codegen{Query: parent}, nil
}

func (s *codegenSchema) localCache(ctx context.Context, parent *core.Engine, args struct{}) (*core.EngineCache, error) {
	if err := parent.Query.RequireMainClient(ctx); err != nil {
		return nil, err
	}
	return &core.EngineCache{
		Query:     parent.Query,
		KeepBytes: int(parent.Query.EngineLocalCacheKeepBytes()),
	}, nil
}

func (s *codegenSchema) cacheEntrySet(ctx context.Context, parent *core.EngineCache, args struct{}) (inst dagql.Instance[*core.EngineCacheEntrySet], _ error) {
	if err := parent.Query.RequireMainClient(ctx); err != nil {
		return inst, err
	}

	entrySetMap, err := parent.Query.EngineCacheEntrySetMap(ctx)
	if err != nil {
		return inst, fmt.Errorf("failed to load cache entry set map: %w", err)
	}

	entrySet, err := parent.Query.EngineLocalCacheEntries(ctx)
	if err != nil {
		return inst, fmt.Errorf("failed to load cache entries: %w", err)
	}

	//
	id := identity.NewID()
	entrySetMap.Store(id, entrySet)
	err = s.srv.Select(ctx, s.srv.Root(), &inst,
		dagql.Selector{
			Field: "__internalCacheEntrySet",
			Args: []dagql.NamedInput{
				{
					Name:  "cacheEntrySetId",
					Value: dagql.NewString(id),
				},
			},
		},
	)
	if err != nil {
		return inst, fmt.Errorf("failed to select cache entry set: %w", err)
	}
	return inst, nil
}

func (s *codegenSchema) cachePrune(ctx context.Context, parent *core.EngineCache, args struct{}) (dagql.Nullable[core.Void], error) {
	void := dagql.Null[core.Void]()
	if err := parent.Query.RequireMainClient(ctx); err != nil {
		return void, err
	}

	_, err := parent.Query.PruneEngineLocalCacheEntries(ctx)
	if err != nil {
		return void, fmt.Errorf("failed to prune cache entries: %w", err)
	}

	return void, nil
}

func (s *codegenSchema) internalCacheEntrySet(ctx context.Context, parent *core.Query, args struct {
	CacheEntrySetID string
},
) (*core.EngineCacheEntrySet, error) {
	if err := parent.RequireMainClient(ctx); err != nil {
		return nil, err
	}

	entrySetMap, err := parent.EngineCacheEntrySetMap(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to load cache entry set map: %w", err)
	}

	v, ok := entrySetMap.Load(args.CacheEntrySetID)
	if !ok {
		return nil, fmt.Errorf("cache entry set not found: %q", args.CacheEntrySetID)
	}
	entrySet, ok := v.(*core.EngineCacheEntrySet)
	if !ok {
		return nil, fmt.Errorf("invalid cache entry set type: %T", v)
	}
	return entrySet, nil
}

func (s *codegenSchema) cacheEntrySetEntries(ctx context.Context, parent *core.EngineCacheEntrySet, args struct{}) ([]*core.EngineCacheEntry, error) {
	return parent.EntriesList, nil
}

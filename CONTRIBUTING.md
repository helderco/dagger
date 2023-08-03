The best way to find a good contribution is to use Dagger for something. Then write down what problems you encounter.
Could be as simple as a question you had, that the docs didn't answer. Or a bug in the tool, or a missing feature.
Then pick an item that you're comfortable with in terms of difficulty, and give it a try. 🙂

You can ask questions along the way, we're always happy to help you with your contribution. The bigger the contribution,
the earlier you should talk to maintainers to make sure you're taking the right approach and are not wasting your effort
on something that will not get merged.

## GitHub Workflow

The recommended workflow is to fork the repository and open pull requests from your fork.

### 1. Fork, clone & configure Dagger upstream

- Click on the _Fork_ button on GitHub
- Clone your fork
- Add the upstream repository as a new remote

```shell
# Clone repository
git clone https://github.com/$YOUR_GITHUB_USER/$REPOSITORY.git

# Add upstream origin
git remote add upstream git@github.com:dagger/$REPOSITORY.git
```

### 2. Create a pull request

```shell
# Create a new feature branch
git checkout -b my_feature_branch

# Make changes to your branch
# ...

# Commit changes - remember to sign!
git commit -s

# Push your new feature branch
git push my_feature_branch

# Create a new pull request from https://github.com/dagger/$REPOSITORY
```

### 3. Add release notes fragment

If this is a user-facing change, please add a line for the release notes.
You will need to have [`changie` installed](https://changie.dev/guide/installation/).

If this is a user-facing change in the 🚙 Engine or 🚗 CLI, run `changie new` in the top level directory.
Here is an example of what that looks like:

```shell
changie new
✔ Kind … Added
✔ Body … engine: add `Directory.Sync`
✔ GitHub PR … 5414
✔ GitHub Author … helderco
```

If there are code changes in the SDKs, run `changie new` in the corresponding directory, e.g. `sdk/go`, `sdk/nodejs`, etc.

Remember to add & commit the release notes fragment.
This will be used at release time, in the changelog.
Here is an example of the end-result for all release notes fragments: https://github.com/dagger/dagger/blob/v0.6.4/.changes/v0.6.4.md

You can find an asciinema of how `changie` works on https://changie.dev

### 4. Update your pull request with latest changes

```shell
# Checkout main branch
git checkout main

# Update your fork's main branch from upstream
git pull upstream main

# Checkout your feature branch
git checkout my_feature_branch

# Rebase your feature branch changes on top of the updated main branch
git rebase main

# Update your pull request with latest changes
git push -f my_feature_branch
```

## Scope of pull requests

We prefer small incremental changes that can be reviewed and merged quickly.
It's OK if it takes multiple pull requests to close an issue.

The idea is that each improvement should land in Dagger's main branch within a
few hours. The sooner we can get multiple people looking at and agreeing on a
specific change, the quicker we will have it out in a release. The quicker we
can get these small improvementes in a Dagger release, the quicker we can get
feedback from our users and find out what doesn't work, or what we have missed.

The added benefit is that this will force everyone to think about handling
partially implemented features & non-breaking changes. Both are great
approaches, and they work really well in the context of Dagger.

["Small incremental changes ftw"](https://github.com/dagger/dagger/pull/1348#issuecomment-1009628531) -> Small pull requests that get merged within hours!

## Commits

### DCO

Contributions to this project must be accompanied by a Developer Certificate of
Origin (DCO).

All commit messages must contain the Signed-off-by line with an email address
that matches the commit author. When commiting, use the `--signoff` flag:

```shell
git commit -s
```

The Signed-off-by line must match the **author's real name**, otherwise the PR will be rejected.

### Commit messages

:::tip
[How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
:::

Guidelines:

- **Group Commits:** Each commit should represent a meaningful change (e.g. implement feature X, fix bug Y, ...).
  - For instance, a PR should not look like _1) Add Feature X 2) Fix Typo 3) Changes to features X 5) Bugfix for feature X 6) Fix Linter 7)_ ...
  - Instead, these commits should be squashed together into a single "Add Feature" commit.
- Each commit should work on its own: it must compile, pass the linter and so on.
  - This makes life much easier when using `git log`, `git blame`, `git bisect`, etc.
  - For instance, when doing a `git blame` on a file to figure out why a change
  was introduced, it's pretty meaningless to see a _Fix linter_ commit message.
  "Add Feature X" is much more meaningful.
- Use `git rebase -i main` to group commits together and rewrite their commit message.
- To add changes to the previous commit, use `git commit --amend -s`. This will
  change the last commit (amend) instead of creating a new commit.
- Format: Use the imperative mood in the subject line: "If applied, this commit
  will _your subject line here_"
- Add the following prefixes to your commit message to help trigger automated processes[^1]:
  - `docs:` for documentation changes only (e.g., `docs: Fix typo in X`);
  - `test:` for changes to tests only (e.g., `test: Check if X does Y`);
  - `chore:` general things that should be excluded (e.g., `chore: Clean up X`);
  - `website:` for the documentation website (i.e., the frontend code; e.g., `website: Add X link to navbar`);
  - `ci:` for internal CI specific changes (e.g., `ci: Enable X for tests`);
  - `infra:` for infrastructure changes (e.g., `infra: Enable cloudfront for X`);
  - `fix`:  for improvements and bugfixes that do not introduce a feature (e.g., `fix: improve error message`);
  - `feat`: for new features (e.g., `feat: implement --cache-to feature to export cache`)

[^1]: See [https://www.conventionalcommits.org](https://www.conventionalcommits.org)

## Docs

### Use relative file paths for links

Instead of using URLs to link to a doc page, use relative file paths instead:

```markdown
❌ This is [a problematic link](/doc-url).

✅ This is [a good link](../relative-doc-file-path.md).
```

The docs compiler will replace file links with URLs automatically. This helps
prevent broken internal links. If a file gets renamed, the compiler will catch
broken links and throw an error. [Learn
more](https://docusaurus.io/docs/markdown-features/links).

## FAQ

### How to run linters locally?

To run all linters:

```shell
./hack/make lint
```

To list available linters:

```shell
> ./hack/make -l | grep lint
docs:lint              lints documentation files
engine:lint            lints the engine
lint                   runs all linters
sdk:all:lint           runs all SDK linters
sdk:go:lint            lints the Go SDK
sdk:nodejs:lint        lints the Node.js SDK
sdk:python:lint        lints the Python SDK
```

:::tip
The `docs:lint` is misleading as it only lints the Markdown in documentation (`.md`). Go snippets in documentation are linted in `engine:lint` while the others are linted in `sdk:<name>:lint`.
:::

### How to re-run all GitHub Actions jobs?

There isn't a button that Dagger contributors can click in their fork that will
re-run all GitHub Actions jobs. See issue
[#1169](https://github.com/dagger/dagger/issues/1169) for more context.

The current workaround is to re-create the last commit:

```shell
git commit --amend -s

# Force push the new commit to re-run all GitHub Actions jobs:
git push -f mybranch
```
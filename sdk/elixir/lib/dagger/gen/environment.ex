# This file generated by `mix dagger.gen`. Please DO NOT EDIT.
defmodule Dagger.Environment do
  @moduledoc "A group of Dagger entrypoints that can be queried and/or invoked."
  use Dagger.QueryBuilder
  @type t() :: %__MODULE__{}
  defstruct [:selection, :client]

  (
    @doc "The check in this environment with the given name, if any\n\n## Required Arguments\n\n* `name` -"
    @spec check(t(), Dagger.String.t()) :: {:ok, Dagger.Check.t() | nil} | {:error, term()}
    def check(%__MODULE__{} = environment, name) do
      selection = select(environment.selection, "check")
      selection = arg(selection, "name", name)

      case execute(selection, environment.client) do
        {:ok, nil} -> {:ok, nil}
        {:ok, data} -> Nestru.decode_from_map(data, Dagger.Check)
        error -> error
      end
    end
  )

  (
    @doc "The list of checks in this environment"
    @spec checks(t()) :: {:ok, [Dagger.Check.t()] | nil} | {:error, term()}
    def checks(%__MODULE__{} = environment) do
      selection = select(environment.selection, "checks")
      execute(selection, environment.client)
    end
  )

  (
    @doc "A unique identifier for this environment."
    @spec id(t()) :: {:ok, Dagger.EnvironmentID.t()} | {:error, term()}
    def id(%__MODULE__{} = environment) do
      selection = select(environment.selection, "id")
      execute(selection, environment.client)
    end
  )

  (
    @doc "Initialize this environment from its source. The full context needed to execute\nthe environment is provided as environmentDirectory, with the environment's configuration\nfile located at configPath.\n\n## Required Arguments\n\n* `environment_directory` - \n* `config_path` -"
    @spec load(t(), Dagger.Directory.t(), Dagger.String.t()) :: Dagger.Environment.t()
    def load(%__MODULE__{} = environment, environment_directory, config_path) do
      selection = select(environment.selection, "load")

      (
        {:ok, id} = Dagger.Directory.id(environment_directory)
        selection = arg(selection, "environmentDirectory", id)
      )

      selection = arg(selection, "configPath", config_path)
      %Dagger.Environment{selection: selection, client: environment.client}
    end
  )

  (
    @doc "Name of the environment"
    @spec name(t()) :: {:ok, Dagger.String.t()} | {:error, term()}
    def name(%__MODULE__{} = environment) do
      selection = select(environment.selection, "name")
      execute(selection, environment.client)
    end
  )

  (
    @doc "This environment plus the given check\n\n## Required Arguments\n\n* `id` -"
    @spec with_check(t(), Dagger.Check.t()) :: Dagger.Environment.t()
    def with_check(%__MODULE__{} = environment, id) do
      selection = select(environment.selection, "withCheck")
      selection = arg(selection, "id", id)
      %Dagger.Environment{selection: selection, client: environment.client}
    end
  )

  (
    @doc "This environment with the given workdir\n\n## Required Arguments\n\n* `workdir` -"
    @spec with_workdir(t(), Dagger.Directory.t()) :: Dagger.Environment.t()
    def with_workdir(%__MODULE__{} = environment, workdir) do
      selection = select(environment.selection, "withWorkdir")

      (
        {:ok, id} = Dagger.Directory.id(workdir)
        selection = arg(selection, "workdir", id)
      )

      %Dagger.Environment{selection: selection, client: environment.client}
    end
  )

  (
    @doc "The directory the environment code will execute in as its current working directory."
    @spec workdir(t()) :: {:ok, Dagger.DirectoryID.t()} | {:error, term()}
    def workdir(%__MODULE__{} = environment) do
      selection = select(environment.selection, "workdir")
      execute(selection, environment.client)
    end
  )
end
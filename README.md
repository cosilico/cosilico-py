# `skeleton-tool`

A skeleton for creating Python tools. Includes build, documentation, and publishing steps.

**Usage**:

```console
$ skeleton-tool [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `config`: Configure the skeleton tool
* `load`: Load the portal gun
* `login`: Login to skeleton tool
* `shoot`: Shoot the portal gun

## `skeleton-tool config`

Configure the skeleton tool

**Usage**:

```console
$ skeleton-tool config [OPTIONS]
```

**Options**:

* `--organization TEXT`: this is a help message  [required]
* `--username TEXT`: [required]
* `--password TEXT`: [required]
* `--make-default / --no-make-default`: [required]
* `--help`: Show this message and exit.

## `skeleton-tool load`

Load the portal gun

**Usage**:

```console
$ skeleton-tool load [OPTIONS] NAME [FOO]
```

**Arguments**:

* `NAME`: name of blah  [required]
* `[FOO]`: blah blah

**Options**:

* `--bar INTEGER`: foo foo bar  [default: 1]
* `--formal / --no-formal`: [default: no-formal]
* `--help`: Show this message and exit.

## `skeleton-tool login`

Login to skeleton tool

**Usage**:

```console
$ skeleton-tool login [OPTIONS]
```

**Options**:

* `--username TEXT`
* `--password TEXT`
* `--organization TEXT`: [default: default]
* `--help`: Show this message and exit.

## `skeleton-tool shoot`

Shoot the portal gun

**Usage**:

```console
$ skeleton-tool shoot [OPTIONS] NAME
```

**Arguments**:

* `NAME`: [required]

**Options**:

* `--help`: Show this message and exit.


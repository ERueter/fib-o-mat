import pathlib
import itertools
import textwrap


# https://stackoverflow.com/a/7440332
def removeprefix(text, prefix):
    return text[len(prefix) :] if text.startswith(prefix) else text


root = pathlib.Path("src/fibomat")

bases = [root]

while bases:
    base = bases.pop()

    current_files = []

    children = []

    for path in sorted(
        # itertools.chain(base.glob("*.py"), base.glob("*.typed"), base.glob("*.pyi"))
        base.glob("*")
    ):
        if path.is_file() and path.name not in ["meson.build", ".DS_Store"]:
            current_files.append(path.name)
        elif path.is_dir():
            bases.append(path)
            children.append(path.name)

    print(removeprefix(str(base), "src/"))

    meson_file = textwrap.dedent(
        """\
        python_sources = [
        {}
        ]

        py.install_sources(
            python_sources,
            subdir: '{}'
        )

        {}
    """
    ).format(
        textwrap.indent(",\n".join(map(lambda x: f"'{x}'", current_files)), "    "),
        removeprefix(str(base), "src/"),
        "\n".join(map(lambda x: f"subdir('{x}')", children)),
    )

    with open(base / "meson.build", "w") as fp:
        fp.write(meson_file)

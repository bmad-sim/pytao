#!/usr/bin/env python
# coding: utf-8

import datetime
import json
import keyword
import os
import re
import shutil
import sys

from numpydoc.docscrape import NumpyDocString

from pytao.util import parsers

CMDS_OUTPUT = "./pytao/interface_commands.py"
TEST_OUTPUT = "./pytao/tests/test_interface_commands.py"

tao_docs = os.path.join(os.getenv("ACC_ROOT_DIR", "../bmad"), "tao", "doc")


def sanitize_method_name(method: str) -> str:
    clean_name = method.replace(":", "_")
    if clean_name == "global":
        clean_name = "tao_global"
    if clean_name in keyword.kwlist:
        clean_name = clean_name + "_"
    return clean_name.strip()


def sanitize(text: str) -> str:
    if "!" in text:
        ex_pos = text.find("!")
        text = text[:ex_pos]
    return text.replace("%", "_").replace("(", "_").replace(")", "").replace("?", "").strip()


def add_tabs(text: str, tabs: int) -> str:
    return "    " * tabs + text.replace("\n", "\n" + "    " * tabs)


def sanitize_help_section(text: str) -> str:
    # Pattern to match \{text\} where text is any character except '\'
    # Using non-greedy matching with .*? to handle nested tags properly
    pattern = r"\\{(.*?)\\}"
    return re.sub(pattern, r"\1", text)


def get_usage_from_help_section(section: str, latex_source: str) -> list[str]:
    complex_commands = ["write", "set", "create"]

    if section in complex_commands:
        examples: list[str] = []
        example_pattern = re.compile(r"\\begin{example}(.*?)\\end{example}", re.DOTALL)

        for match in list(example_pattern.finditer(latex_source))[1:]:
            # Skip the first match as it's typically the format
            example_text = match.group(1).strip()
            examples.extend(example_text.splitlines())
        return examples

    usage_match = re.search(r"\\begin{example}(.*?)\\end{example}", latex_source, re.DOTALL)

    if usage_match:
        format_text = [line.strip() for line in usage_match.group(1).strip().splitlines()]
        return format_text
    return [section]


def parse_command_list_help(command_list_tex: str):
    with open(command_list_tex) as fp:
        contents = fp.read()

    sections = {}
    section = None
    for line in contents.splitlines():
        if line.startswith(r"\section{") and "commands!" in line:
            section = line.split("{")[1].split("}")[0]
            assert section not in sections
            sections[section] = []
        if section:
            sections[section].append(line)

    def split_comment(line: str) -> tuple[str, str]:
        line = line.strip()
        if "!" in line:
            cmd, comment = line.split("!", 1)
        else:
            cmd, comment = line, ""
        return cmd.strip(), comment.strip()

    return {
        section: [
            split_comment(sanitize_help_section(line))
            for line in get_usage_from_help_section(section, "\n".join(lines))
            if line.startswith(section)
        ]
        for section, lines in sections.items()
    }


def generate_autocompletion(command_list_tex: str):
    return parse_command_list_help(command_list_tex)


def generate_params(params):
    """
    Generates the list of parameters for the Tao Python method.

    This method uses the NumpyDocString Parameter class to introspect for
    optional flags.

    `verbose` and `as_dict`, `raises` are always keyword arguments defaulting to True`.

    Parameters
    ----------
    params : list
      List of Parameter objects obtained via parsing the Tao docstring with NumpyDocString.

    Returns
    -------
    strq
       The list of arguments properly formatted.
       E.g.: tao, s, *, ix_uni="1", ix_branch="0", which="model", verbose=False, as_dict=True
    """

    args = ["self"]
    kwargs = []
    for param in params:
        name = sanitize(param.name)

        # Skip empty params.
        if not name:
            assert len(params) == 1
            continue

        dtype = param.type
        if "default=" in dtype:
            kwargs.append(f"{name}='{dtype[dtype.find('=') + 1 :].strip()}'")
        elif "optional" in dtype:
            kwargs.append(f"{name}=''")
        else:
            args.append(name)

    kwargs.append("verbose=False")
    kwargs.append("as_dict=True")
    kwargs.append("raises=True")

    return ", ".join(args + ["*"] + kwargs)


def generate_method_code(command_str, docs, method, command, returns):
    """
    Generates the Python code to execute the Tao method.
    This function relies on a specific annotation on the Returns block of the docstring
    so that the proper data type can be returned.

    Parameters
    ----------
    docs : NumpyDocString
      The NumpyDocString instance
    method : str
      The cleaned method name
    command : str
      The `command_str` text from the JSON parser. This is a Python f-string for the Tao command.
      E.g.: "python lat_list {flags} {ix_uni}_{ix_branch}>>{elements}|{which} {who}"
    returns : list
      List of Parameter objects obtained via parsing the Tao docstring with NumpyDocString.

    Returns
    -------
    str
       The list of arguments properly formatted.
       E.g.: tao, *, flags="", ix_uni, ix_branch, elements, which, who, verbose=True, as_dict=True, raises=True
    """
    code_list = [f"cmd = f'{command_str}'"]
    code_list.append("if verbose: print(cmd)")
    for r in returns:
        tp = "string_list"
        if r.type and "??" not in r.type:
            tp = r.type
        if not len(r.desc):
            # No conditionals for code execution
            special_parser = getattr(parsers, f"parse_{method}", "")
            if special_parser:
                parser_docs = NumpyDocString(special_parser.__doc__)
                docs["Returns"] = parser_docs["Returns"]
            code_list.append(
                f"return self.__execute(cmd, as_dict, raises, method_name='{method}', cmd_type='{tp}')"
            )
        else:
            code_list.append(
                f"{r.desc[0]}:\n    return self.__execute(cmd, as_dict, raises, method_name='{method}', cmd_type='{tp}')"
            )
    return "\n".join(code_list)


autogenerated_header = f"""\
# ==============================================================================
# AUTOGENERATED FILE - DO NOT MODIFY
# This file was generated by the script `generate_interface_commands.py`.
# Any modifications may be overwritten.
# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ==============================================================================
"""


def read_interface_commands():
    # Read the JSON File
    for command_name in ("pipe", "python"):
        f_name = os.path.join(tao_docs, f"{command_name}-interface-commands.json")
        if os.path.exists(f_name):
            print(f"Reading JSON from: {f_name}")
            with open(f_name, "r") as f:
                return json.load(f)

    print(
        f"Unable to find an interface commands JSON file in path: {tao_docs})",
        file=sys.stderr,
    )
    exit(1)


def read_interface_template():
    with open("pytao/interface.tpl.py", "r") as f:
        return f.read()


def write_interface_commands():
    interface_tpl_py = read_interface_template()
    cmds_from_tao = read_interface_commands()
    cmds_to_module = [
        autogenerated_header,
        interface_tpl_py,
    ]

    print()

    hotfixes = {}

    for method, metadata in cmds_from_tao.items():
        if method in hotfixes:
            metadata.update(hotfixes[method])

        docstring = metadata["description"]
        command_str = sanitize(metadata["command_str"])

        clean_method = sanitize_method_name(method)
        np_docs = NumpyDocString(docstring)

        params = generate_params(np_docs["Parameters"])
        try:
            code = generate_method_code(
                command_str,
                np_docs,
                clean_method,
                command_str,
                np_docs["Returns"],
            )
        except Exception as ex:
            print(f"***Error generating code for: {method}. Exception was: {ex}")
            raise

        method_template = f"""
    def {clean_method}({params}):
{add_tabs('"""', 2)}
{add_tabs(str(np_docs), 2)}
{add_tabs('"""', 2)}
{add_tabs(code, 2)}

    """
        cmds_to_module.append(method_template)

    command_to_usage = generate_autocompletion(os.path.join(tao_docs, "command-list.tex"))
    cmds_to_module.append(f"_autocomplete_usage_ = {command_to_usage!r}")

    with open(CMDS_OUTPUT, "w") as out:
        out.writelines(cmds_to_module)

    print(f"Generated file: {CMDS_OUTPUT}")


def get_tests(examples):
    """
    Parse examples to extract test cases with their initialization and arguments.

    Parameters
    ----------
    examples : list of str
        A list of strings containing test examples in a specific format.
        The format should follow these patterns:
        - "Example: <test_name>" to start a new test case
        - "init: <initialization_code>" to specify initialization code
        - "args:" to indicate the start of arguments list
        - "<arg_name>: <arg_value>" for each argument

    Returns
    -------
    dict
        A dictionary where keys are test names and values are dictionaries with:
        - 'init': initialization code (string)
        - 'args': dictionary of argument names to their values
    """
    tests = {}
    name = ""
    parsing_args = False
    for ex in examples:
        if not ex:
            continue
        anchor = ex.find(":")
        if "xample:" in ex:
            parsing_args = False
            name = ex[anchor + 1 :].strip()
            tests[name] = {}
            continue
        if "init:" in ex:
            tests[name]["init"] = ex[anchor + 1 :].strip()
            continue
        if "args:" in ex:
            parsing_args = True
            tests[name]["args"] = {}
            continue
        if parsing_args:
            arg_name = ex[:anchor].strip()
            arg_value = ex[anchor + 1 :].strip()
            tests[name]["args"][arg_name] = arg_value
    return tests


def write_tests():
    cmds_from_tao = read_interface_commands()
    cmds_to_test_module = [
        autogenerated_header,
        "from .conftest import ensure_successful_parsing, new_tao",
    ]

    for method, metadata in cmds_from_tao.items():
        clean_method = sanitize_method_name(method)
        docstring = metadata["description"]
        np_docs = NumpyDocString(docstring)

        examples = np_docs["Examples"]
        tests = get_tests(examples)

        if len(tests) == 0:
            print(f"No examples found for: {method}")

        for test_name, test_meta in tests.items():
            args = [f"{k}='{v}'" for k, v in test_meta["args"].items()]
            args.append("verbose=True")
            test_code = f"""
with ensure_successful_parsing(caplog):
    with new_tao(tao_cls, '{test_meta["init"]}', external_plotting=False) as tao:
        tao.{clean_method}({", ".join(args)})
            """
            method_template = f"""
def test_{clean_method}_{test_name}(caplog, tao_cls):
{add_tabs(test_code, 1)}
            """
            cmds_to_test_module.append(method_template)

    with open(TEST_OUTPUT, "w") as out:
        out.writelines(cmds_to_test_module)

    print(f"Generated file: {TEST_OUTPUT}")

    if shutil.which("ruff"):
        os.system(f'ruff format "{CMDS_OUTPUT}" "{TEST_OUTPUT}"')
        os.system(f'ruff check --fix "{CMDS_OUTPUT}" "{TEST_OUTPUT}"')


if __name__ == "__main__":
    write_interface_commands()
    write_tests()

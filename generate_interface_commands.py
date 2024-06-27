#!/usr/bin/env python
# coding: utf-8

from numpydoc.docscrape import NumpyDocString
from pytao.util import parsers
import datetime
import json
import keyword
import os
import shutil

CMDS_OUTPUT = "./pytao/interface_commands.py"
TEST_OUTPUT = "./pytao/tests/test_interface_commands.py"

# ## Read the JSON File
f_name = f'{os.getenv("ACC_ROOT_DIR")}/tao/doc/python-interface-commands.json'
print(f'Reading JSON from: {f_name}')

with open(f_name, 'r') as f:
    cmds_from_tao = json.load(f)

with open("interface.tpl.py", 'r') as f:
    interface_tpl_py = f.read()

# ### Utilitary Functions
def sanitize_method_name(method):
    clean_name = method.replace(':', '_')
    if clean_name == "global":
        clean_name = "tao_global"
    if clean_name in keyword.kwlist:
        clean_name = clean_name+"_"
    return clean_name.strip()


def sanitize(text):
    if '!' in text:
        ex_pos = text.find('!')
        text = text[:ex_pos]
    return text.replace('%', '_').replace('(', '_').replace(')', '').replace('?', '').strip()


def add_tabs(text, tabs):
    return '    '*tabs+text.replace('\n', '\n'+'    '*tabs)


# ### Key Functions for Code Generation
def generate_params(params):
    """
    Generates the list of parameters for the Tao Python method.
    This method uses the NumpyDocString Parameter class to introspect
    for optional flags.
        
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

    args = ['self']
    kwargs = []
    for idx, p in enumerate(params):
        name = sanitize(p.name)
        
        # Skip empty params. 
        if not name:
            assert len(params) == 1
            continue
        
        dtype = p.type
        if 'default=' in dtype:
            kwargs.append(f"{name}='{dtype[dtype.find('=')+1:].strip()}'")
        elif 'optional' in dtype:
            kwargs.append(f"{name}=''")
        else:
            args.append(name)

    kwargs.append('verbose=False')
    kwargs.append('as_dict=True')
    kwargs.append('raises=True')
    
    param_str =  ', '.join(args + ['*'] + kwargs)

    return param_str


def generate_method_code(docs, method, command, returns):
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
        tp = 'string_list'
        if r.type and '??' not in r.type:
            tp = r.type
        if not len(r.desc):
            # No conditionals for code execution
            special_parser = getattr(parsers, f'parse_{method}', "")
            if special_parser:
                parser_docs = NumpyDocString(special_parser.__doc__)
                docs['Returns'] = parser_docs['Returns']
            code_list.append(f"return self.__execute(cmd, as_dict, raises, method_name='{method}', cmd_type='{tp}')")
        else:
            code_list.append(f"{r.desc[0]}:\n    return self.__execute(cmd, as_dict, raises, method_name='{method}', cmd_type='{tp}')")
    return '\n'.join(code_list)


# ## Parse the JSON Dictionary and Write the Python module

cmds_to_module = [
    f'''# ==============================================================================
# AUTOGENERATED FILE - DO NOT MODIFY
# This file was generated by the script `generate_interface_commands.py`.
# Any modifications may be overwritten.
# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ==============================================================================

{interface_tpl_py}
'''
]

print()


# TODO: bring these back to bmad
hotfixes = {
    "var": {
        "command_str": "python var {var} {slaves}",
    }
}

for method, metadata in cmds_from_tao.items():
    if method in hotfixes:
        metadata.update(hotfixes[method])

    docstring = metadata['description']
    command_str = sanitize(metadata['command_str'])

    clean_method = sanitize_method_name(method)
    np_docs = NumpyDocString(docstring)

    params = generate_params(np_docs['Parameters'])
    try:
        code = generate_method_code(np_docs, clean_method, command_str, np_docs['Returns'])
    except Exception as ex:
        print(f'***Error generating code for: {method}. Exception was: {ex}')
        raise

    method_template = f'''
    def {clean_method}({params}):
{add_tabs('"""', 2)}
{add_tabs(str(np_docs), 2)}
{add_tabs('"""', 2)}
{add_tabs(code, 2)}

'''
    cmds_to_module.append(method_template)
    
with open(CMDS_OUTPUT, 'w') as out:
    out.writelines(cmds_to_module)

print(f'Generated file: {CMDS_OUTPUT}')

# ## Parse the JSON Dictionary and Write the Python Test module

def get_tests(examples):
    tests = {}
    name = ''
    parsing_args = False
    for ex in examples:
        if not ex:
            continue
        anchor = ex.find(':')
        if 'xample:' in ex:
            parsing_args = False
            name = ex[anchor+1:].strip()
            tests[name] = {}
            continue
        if 'init:' in ex:
            tests[name]['init'] = ex[anchor+1:].strip()
            continue
        if 'args:' in ex:
            parsing_args = True
            tests[name]['args'] = {}
            continue
        if parsing_args:
            arg_name = ex[:anchor].strip()
            arg_value = ex[anchor+1:].strip()
            tests[name]['args'][arg_name] = arg_value
    return tests

cmds_to_test_module = [f"""# ==============================================================================
# AUTOGENERATED FILE - DO NOT MODIFY
# This file was generated by the script `generate_interface_commands.py`.
# Any modifications may be overwritten.
# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ==============================================================================

from .conftest import ensure_successful_parsing, new_tao

"""]

for method, metadata in cmds_from_tao.items():
    clean_method = sanitize_method_name(method)
    docstring = metadata['description']
    np_docs = NumpyDocString(docstring)
    
    examples = np_docs['Examples']
    tests = get_tests(examples)
    
    if len(tests) == 0:
        print(f'No examples found for: {method}')
    
    for test_name, test_meta in tests.items():
        args = [f"{k}='{v}'" for k, v in test_meta['args'].items()]
        args.append("verbose=True")
        test_code = f'''
with ensure_successful_parsing(caplog):
    with new_tao(tao_cls, '{test_meta['init']}') as tao:
        tao.{clean_method}({', '.join(args)})
        '''
        method_template = f'''
def test_{clean_method}_{test_name}(caplog, tao_cls):
{add_tabs(test_code, 1)}
        '''
        cmds_to_test_module.append(method_template)

with open(TEST_OUTPUT, 'w') as out:
    out.writelines(cmds_to_test_module)

print(f'Generated file: {TEST_OUTPUT}')

if shutil.which("ruff"):
    os.system(f'ruff format "{CMDS_OUTPUT}" "{TEST_OUTPUT}"')

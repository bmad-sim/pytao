#!/usr/bin/env python
# coding: utf-8

import os
import json
import pprint
import keyword
from numpydoc.docscrape import NumpyDocString

CMDS_OUTPUT = "./pytao/interface_commands.py"
TEST_OUTPUT = "./pytao/tests/test_interface_commands.py"

# ## Read the JSON File
f_name = f'{os.getenv("ACC_ROOT_DIR")}/tao/doc/python-interface-commands.json'
print(f'Reading JSON from: {f_name}')

with open(f_name, 'r') as f:
    cmds_from_tao = json.load(f)


# ### Utilitary Functions
def sanitize_method_name(method):
    clean_name = method.replace(':', '_')
    if clean_name == "global":
        clean_name = "globals"
    if clean_name in keyword.kwlist:
        clean_name = clean_name+"_"
    return clean_name.strip()


def sanitize(text):
    if '!' in text:
        ex_pos = text.find('!')
        text = text[:ex_pos]
    return text.replace('@', '_').replace('%', '_').replace('(', '_').replace(')', '').replace('?', '').strip()


def add_tabs(text, tabs):
    return '    '*tabs+text.replace('\n', '\n'+'    '*tabs)


# ### Key Functions for Code Generation
def generate_params(params):
    """
    Generates the list of parameters for the Tao Python method.
    This method uses the NumpyDocString Parameter class to introspect
    for optional flags.
    
    If the Tao method has more than 2 parameters this function make so that they must be
    keyword arguments otherwise they are positional arguments.
    
    `verbose` and `as_dict` are always keyword arguments defaulting to True`.
    
    Parameters
    ----------
    params : list
      List of Parameter objects obtained via parsing the Tao docstring with NumpyDocString.
    
    Returns
    -------
    str
       The list of arguments properly formatted.
       E.g.: tao, *, flags="", ix_uni, ix_branch, elements, which, who, verbose=True, as_dict=True
    """
    param_list = ['tao']
    if len(params) > 2:
        param_list.append('*')
    for p in params:
        name = sanitize(p.name)
        dtype = p.type
        optional = '=""' if 'optional' in dtype else ''
        default = f'="{dtype[dtype.find("=")+1:].strip()}"' if 'default=' in dtype else ''
        extra = optional if not default else default
        if extra and len(params) <= 2 and '*' not in param_list:
            param_list.insert(1, '*')
        param_list.append(f'{name}{extra}')
        
    if len(params) <= 2 and '*' not in param_list:
        param_list.append('*')
    param_list.append('verbose=False')
    param_list.append('as_dict=True')
    return ', '.join(param_list)


def generate_method_code(command, returns):
    """
    Generates the Python code to execute the Tao method.
    This function relies on a specific annotation on the Returns block of the docstring
    so that the proper data type can be returned.
     
    Parameters
    ----------
    command : str
      The `command_str` text from the JSON parser. This is a Python f-string for the Tao command.
      E.g.: "python lat_list {flags} {ix_uni}_{ix_branch}>>{elements}|{which} {who}"
      
    returns : list
      List of Parameter objects obtained via parsing the Tao docstring with NumpyDocString.
    
    Returns
    -------
    str
       The list of arguments properly formatted.
       E.g.: tao, *, flags="", ix_uni, ix_branch, elements, which, who, verbose=True, as_dict=True
    """
    method_for_type = {
        'None': '__exec_no_return',
        'integer_array': '__exec_integer',
        'real_array': '__exec_real',
        'string_list': '__exec_string'
    }
    code_list = [f"cmd = f'{command_str}'"]
    code_list.append("if verbose: print(cmd)")
    for r in returns:
        if not len(r.desc):
            tp = 'string_list'
            if r.type and '??' not in r.type:
                tp = r.type
            code_list.append(f"return {method_for_type[tp]}(tao, cmd, as_dict)")
        else:
            code_list.append(f"{r.desc[0]}:\n    return {method_for_type[r.type]}(tao, cmd, as_dict)")
    return '\n'.join(code_list)


# ## Parse the JSON Dictionary and Write the Python module

cmds_to_module = ["""
from pytao.tao_ctypes.util import parse_tao_python_data
from pytao.util.parameters import tao_parameter_dict


def __exec_no_return(tao, cmd, as_dict=True):
    tao.cmd(cmd)
    return None


def __exec_string(tao, cmd, as_dict=True):
    ret = tao.cmd(cmd)
    try:
        if as_dict:
            data = parse_tao_python_data(ret)
        else:
            data = tao_parameter_dict(ret)
    except:
        data = ret
    return data
    

def __exec_integer(tao, cmd, as_dict=True):
    ret = tao.cmd_integer(cmd)
    return ret
    
    
def __exec_real(tao, cmd, as_dict=True):
    ret = tao.cmd_real(cmd)
    return ret

"""]

for method, metadata in cmds_from_tao.items():
    docstring = metadata['description']
    command_str = sanitize(metadata['command_str'])

    clean_method = sanitize_method_name(method)
    np_docs = NumpyDocString(docstring)

    params = generate_params(np_docs['Parameters'])
    try:
        code = generate_method_code(command_str, np_docs['Returns'])
    except Exception as ex:
        print(f'***Error generating code for: {method}. Exception was: {ex}')

    method_template = f'''
def {clean_method}({params}):
{add_tabs('"""', 1)}
{add_tabs(docstring, 1)}
{add_tabs('"""', 1)}
{add_tabs(code, 1)}

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

cmds_to_test_module = ["""
import os
from pytao import Tao
from pytao import interface_commands

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
        args = ['tao'] + [f"{k}='{v}'" for k, v in test_meta['args'].items()]
        test_code = f'''
tao = Tao(os.path.expandvars('-noplot -init {test_meta['init']}'))
ret = interface_commands.{clean_method}({', '.join(args)})
        '''
        method_template = f'''
def test_{clean_method}_{test_name}():
{add_tabs(test_code, 1)}
        '''
        cmds_to_test_module.append(method_template)

with open(TEST_OUTPUT, 'w') as out:
    out.writelines(cmds_to_test_module)

print(f'Generated file: {TEST_OUTPUT}')
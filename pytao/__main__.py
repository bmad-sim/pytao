import sys


def main():
    try:
        import IPython
    except ImportError as ex:
        print(f"IPython unavailable ({ex}); pytao interactive mode unavailable.")
        exit(1)

    from .interface_commands import Tao

    init_args = " ".join(sys.argv[1:])
    startup_message = f"Initializing `tao` object with the following: {init_args}"
    print("-" * len(startup_message))
    print(startup_message)
    print()
    print("Type`tao.` and hit tab to see available commands.")
    print("-" * len(startup_message))
    print()

    tao = Tao(init=init_args)
    return IPython.start_ipython(user_ns={"tao": tao}, argv=[])


if __name__ == "__main__":
    main()

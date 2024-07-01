from .main import tao_root_window


def main():
    root = tao_root_window()
    if root.do_mainloop:
        root.mainloop()


if __name__ == "__main__":
    main()

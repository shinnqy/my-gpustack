import argparse
from multiprocessing import freeze_support
from gpustack.cmd import setup_start_cmd


def main():
    parser = argparse.ArgumentParser(
        description="GPUStack",
        conflict_handler="resolve",
        add_help=True,
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=55, indent_increment=2, width=200
        ),
    )
    subparsers = parser.add_subparsers(
        help="sub-command help",
        metavar='{start,reload-config,list-images,save-images,copy-images,load-images,reset-admin-password,version}',
    )

    setup_start_cmd(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        if isinstance(args.func, type):
            args.func(args).run()
        else:
            args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    freeze_support()
    main()

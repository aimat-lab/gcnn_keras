# def run_default(file_path,
#                 nprocs: int = None,
#                 sanitize: bool = True,
#                 add_hydrogen: bool = True,
#                 make_conformers: bool = True,
#                 optimize_conformer: bool = True):
#     if sys.platform[0:3] == 'win':
#         python_command = 'python'  # or 'python.exe'
#     else:
#         python_command = 'python3'
#     script_command = os.path.join(os.path.dirname(os.path.realpath(__file__)), "default.py")
#     command_list = [python_command, script_command,
#                     "--file", file_path,
#                     "--nprocs", str(nprocs),
#                     "--sanitize", str(sanitize),
#                     "--add_hydrogen", str(add_hydrogen),
#                     "--make_conformers", str(make_conformers),
#                     "--optimize_conformer", str(optimize_conformer)
#                     ]
#     return command_list
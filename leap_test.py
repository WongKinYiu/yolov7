from pathlib import Path

from code_loader import LeapLoader


def check_integration():
    dir_path = str(Path(__file__).parent)
    script_path = 'leap_binder.py'
    datascript = LeapLoader(code_path=dir_path, code_entry_name=script_path)
    res = datascript.check_dataset()
    if res.is_valid:
        print("Integration script is valid")
    else:
        print(f"Integration failed with error: {res.general_error}")


if __name__ == '__main__':
    check_integration()
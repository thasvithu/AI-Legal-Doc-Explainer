import sys

def error_message_detail(error):
    _, _, exe_tb = sys.exc_info()
    if exe_tb is None:
        return str(error)
    
    file_name = exe_tb.tb_frame.f_code.co_filename
    line_number = exe_tb.tb_lineno
    return f"Error occurred in Python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)

    def __str__(self):
        return self.error_message

import sys
from src.logger import logger

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(
            error_message, error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        file_name    = exc_tb.tb_frame.f_code.co_filename
        line_number  = exc_tb.tb_lineno
        error_msg = (
            f"Error in script: [{file_name}] "
            f"at line: [{line_number}] "
            f"message: [{str(error_message)}]"
        )
        return error_msg

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return self.__class__.__name__
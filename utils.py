class Utils:
    @classmethod
    def is_bool(cls, str) -> bool:
        return True if str == "True" or str == "False" else False

    @classmethod
    def try_to_bool(cls, str) -> bool | str:
        """文字列をBooleanに変換

        Args:
            str (str): 文字列

        Returns:
            bool | str: 変換できなかった場合は文字列を返す
        """
        if str == "True":
            return True
        if str == "False":
            return False
        return str

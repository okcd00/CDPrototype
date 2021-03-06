# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2022 All rights reserved.
#
#   filename : text_utils.py
#   author   : chendian / okcd00@qq.com
#   date     : 2019-04-08
#   desc     : error detection model.
#              n-gram database for Python2.x
#   reference: functions in Bert's tokenization
# ==========================================================================
from __future__ import absolute_import, division, print_function
import re
import six
import unicodedata
from .encodings import convert_to_unicode


re_han = re.compile("([\u4E00-\u9Fa5a-zA-Z0-9+#&]+)", re.U)
re_skip = re.compile("(\r\n\\s)", re.U)

SPECIAL_CHN_PUNC = {
    '‘': '\'',
    '’': '\'',
    '“': '\"',
    '”': '\"',
    '—': "-",
}


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_to_str(text):
    return printable_text(text)


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((33 <= cp <= 47) or (58 <= cp <= 64) or
        (91 <= cp <= 96) or (123 <= cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


def is_chinese_phrase(phrase_str):
    return False not in list(map(is_chinese_char, map(ord, phrase_str)))


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            if char not in ",，.。":  # this time we consider commas.
                continue  # remove
        if is_whitespace(char):
            output.append(" ")
        else:  # NEW
            output.append(SPECIAL_CHN_PUNC.get(char, char.lower()))
    return "".join(output)


def judge_chinese_mask(tokens, elem_type=bool):
    """return a list of boolean values for chinese char or not"""
    def is_chinese_word(text):
        text = convert_to_unicode(text)
        for char in text:
            cp = ord(convert_to_unicode(char))
            if not is_chinese_char(cp):
                return False if (elem_type is bool) else 0
        return True if (elem_type is bool) else 1

    return list(map(is_chinese_word, tokens))


def split_2_short_text(text, include_symbol=False):
    """
    长句切分为短句
    :param text: str
    :param include_symbol: bool
    :return: (sentence, idx)
    """
    result = []
    blocks = re_han.split(text)
    start_idx = 0
    for blk in blocks:
        if not blk:
            continue
        if include_symbol:
            result.append((blk, start_idx))
        else:
            if re_han.match(blk):
                result.append((blk, start_idx))
        start_idx += len(blk)
    return result


if __name__ == "__main__":
    ret = split_2_short_text("由于具体上市审批事宜需要在本次债券发行结束后方能进行，并依赖于有关主管部门的审批或核准，发行人目前无法保正本次债券一定能够按照预期在上海证券交易所上市流通，且具体上市进程在时间上存在不确定性。")
    print(ret)
from __future__ import print_function
from __future__ import division

import re

filepath = 'war_and_peace.txt'
out_file = 'wap.txt'

prev_line = ''

NEW_LINE_IN_PARAGRAPH_REGEX = re.compile(r'(\S)\r\n(\S)')
MULTIPLE_NEWLINES_REGEX = re.compile(r'(\r\n)(\r\n)+')
R_NEWLINE = re.compile(r'\r')

with open(filepath, 'r') as f_input:
    book_str = f_input.read()


book_str = NEW_LINE_IN_PARAGRAPH_REGEX.sub('\g<1> \g<2>', book_str)
book_str = MULTIPLE_NEWLINES_REGEX.sub('\n\n', book_str)
book_str = R_NEWLINE.sub('', book_str)

with open(out_file, 'w') as f_output:
    f_output.write(book_str)

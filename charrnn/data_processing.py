from __future__ import print_function
from __future__ import division

import re
import codecs

filepath = 'war_and_peace.txt'
out_file = 'wap.txt'

prev_line = ''

NEW_LINE_IN_PARAGRAPH_REGEX = re.compile(r'(\S)\n(\S)')
MULTIPLE_NEWLINES_REGEX = re.compile(r'(\n)(\n)+')

with codecs.open(filepath, encoding='utf-8', mode='r') as f_input:
    book_str = f_input.read()


book_str = NEW_LINE_IN_PARAGRAPH_REGEX.sub('\g<1> \g<2>', book_str)
book_str = MULTIPLE_NEWLINES_REGEX.sub('\n\n', book_str)

with codecs.open(out_file, encoding='utf-8', mode='w')as f_output:
    f_output.write(book_str)

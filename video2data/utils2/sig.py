#!/usr/bin/python3 
# coding: utf-8
#
# Shot and scene printing

import zlib
import base64
#
#
# Placeholder for eventual hashing algorithm
#  Right now its lossless compression
#
def phash_bits(bitstring):
  if bitstring is not '0b':
    tmp = zlib.compress(bitstring.encode())
    #tmp = int(bitstring, 2).to_bytes((len(bitstring) + 7) // 8, 'big')
    # first 100 
    return base64.urlsafe_b64encode(tmp).decode('utf-8')[:-2][:100]

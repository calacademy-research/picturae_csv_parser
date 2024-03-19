#!/bin/bash
xxd -p test_image.jpg > test_hexdump.txt
xxd -r -p test_hexdump.txt | LC_ALL=C tr -d -c '[:print:]' > ascii_hex.txt


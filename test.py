# data = []
# current_block = []

# with open('gio.csv', 'r') as f:
#     for row in f:
#         if row.startswith('00000000000000'):
#             if current_block:
#                 data.append(' '.join(current_block))
#             current_block = [row.strip()]
#         elif row.startswith('C 1') and current_block:
#             current_block.append(row.strip())
#         # elif current_block:
#         #     current_block.append(row.strip())

#     if current_block:
#         data.append(' '.join(current_block))

# with open('res.txt', 'w') as e:
#     e.write('\n'.join(data))


# data = []

# # Reading and processing data from gio.txt
# with open('gio.txt', 'r') as f:
#     z = 1
#     lines = f.readlines()
#     for i, line in enumerate(lines):
#         if line.startswith('00000000000000'):
#             bit_range = z
#             private_key = line.strip()
#             address_index = i + 1
#             address_line = lines[address_index]
#             if address_line.startswith('C 1'):
#                 address = address_line.strip().split(' ')[1]
#                 data.append(f"{bit_range}, {private_key}, {address}")
#         z += 1

# # Writing the result to res.txt
# with open('res.txt', 'w') as e:
#     e.write('Bit Range, Private Key, Address\n')
#     e.write('\n'.join(data))



data = []

# Reading and processing data from gio.txt
with open('gio.txt', 'r') as f:
    lines = f.readlines()
    bit_range = 1  # Starting the bit range from 1
    for i, line in enumerate(lines):
        if line.startswith('00000000000000'):
            private_key = line.strip()
            address_index = i + 1
            address_line = lines[address_index]
            if address_line.startswith('C 1'):
                address = address_line.strip().split(' ')[1]
                data.append(f"{bit_range}, {private_key}, {address}")
                bit_range += 1  # Incrementing the bit range

# Writing the result to res.txt
with open('res.txt', 'w') as e:
    e.write('Bit Range, Private Key, Address\n')
    e.write('\n'.join(data))

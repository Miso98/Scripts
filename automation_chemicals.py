
def generate_sds_line(chemical_name, cas_number):
    return f"\\addSDS{{{chemical_name} - {cas_number}}}{{SDS/{cas_number}-{chemical_name}-SDS}}\n"
def process_input_from_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(output_file, 'w') as f_out:
        for line in lines:
            # Split by the last occurrence of space to get the chemical name
            last_space_index = line.rfind(' ')
            chemical_name = line[:last_space_index].strip()
            cas_number = line[last_space_index:].strip()
            sds_line = generate_sds_line(chemical_name, cas_number)
            f_out.write(sds_line)
# Replace 'input.txt' and 'output.txt' with your input and output file paths
process_input_from_file('automation_test.txt', 'output.txt')

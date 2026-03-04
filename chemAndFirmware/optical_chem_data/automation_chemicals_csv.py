import csv

def generate_sds_line(chemical_name, cas_number, sds_filename):
    return [chemical_name, cas_number, sds_filename, f"\\addSDS{{{chemical_name} - CAS - {cas_number}}}{{SDS/{sds_filename}.pdf}}"]

def process_input_from_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row if exists
        rows = []
        for i, row in enumerate(reader):
            if i == 0: #skips the first row which contains the chemical name and CAS number
                continue
            chemical_name, cas_number = row
            sds_filename = f"{cas_number}-{chemical_name}-SDS"
            rows.append(generate_sds_line(chemical_name, cas_number, sds_filename))
            #command = f"\\addSDS{{{chemical_name} - {cas_number}}}{{SDS/{sds_filename}.pdf}}\n"
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['chemical name', 'CAS number', 'SDS filename', 'LaTeX code'])
                writer.writerows(rows)

# Replace 'input.csv' and 'output.txt' with your input CSV and output file paths
process_input_from_csv('Chemical_CSV_Test.csv', 'output.csv')

        




    
   
        






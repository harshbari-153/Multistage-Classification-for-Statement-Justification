import csv

# Define the titles to be added
titles = [
    "sr", "id", "label", "statement", "subject(s)", "speaker", 
    "speaker's job title", "state info", "party affiliation", 
    "barely true counts", "false counts", "half true counts", 
    "mostly true counts", "pants on fire counts", "context", "justification"
]

# Read the TSV file and write to CSV file
def convert_tsv_to_csv(tsv_file, csv_file, headers):
    with open(tsv_file, 'r', newline='', encoding='utf-8') as tsv_in, \
         open(csv_file, 'w', newline='', encoding='utf-8') as csv_out:

        tsv_reader = csv.reader(tsv_in, delimiter='\t')
        csv_writer = csv.writer(csv_out)

        # Write the headers first
        csv_writer.writerow(headers)

        # Write the rest of the data
        for row in tsv_reader:
            csv_writer.writerow(row)

# Specify the input and output file names
tsv_file = '..\liar plus dataset\dataset\val2.tsv'
csv_file = '..\liar plus dataset\dataset\val.csv'

# Convert the file
convert_tsv_to_csv(tsv_file, csv_file, titles)

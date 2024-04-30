import os
""" This is to combine all the output smaller chunk files to large text file

 """
# Directory containing the text files
directory = 'Sentiment_Analysis_BertApproach'

# Output file name
output_file = 'Sentiment_Analysis_BertApproach/combined_data_sentiment_bert.csv'

# Open the output file in append mode
with open(output_file, 'a') as outfile:
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            # Open the current file and read its contents
            with open(filepath, 'r') as infile:
                # Write the contents of the current file to the output file
                outfile.write(infile.read())
                # outfile.write('\n')  # Add a newline between files if desired

def convert_iob2_to_conll(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    conll_content = []

    for line in lines:
        if line.startswith('#') or line.strip() == '':
            # Handle document and sentence boundaries, and empty lines
            if line.startswith('# newdoc'):
                conll_content.append("-DOCSTART- -X- -X- O\n")
            elif line.strip() == '':
                conll_content.append("")
            continue
        else:
            # Process token lines
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                token = parts[1]  # The token itself
                ne_tag = parts[2]  # The named entity tag
                
                # Add the converted line to the ConLL content
                conll_line = f"{token} -X- -X- {ne_tag}"
                conll_content.append(conll_line)

    # Write the converted content to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(conll_content))

# Example usage
input_file_path = 'baseline-data/en_ewt-ud-train.iob2'  # Replace with the actual path to your IOB2 file
output_file_path = 'baseline-data/en_ewt-ud-train_CONV.iob2'  # Replace with the desired path for the new CoNLL file

convert_iob2_to_conll(input_file_path, output_file_path)
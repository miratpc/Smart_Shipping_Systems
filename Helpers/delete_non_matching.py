import os

match_to = r"path/to/match/to"

delete_from = r"path/to/delete/from"
delete_from_txt = r"path/to/delete/from/txt"

def delete_non_matching_jpg(delete_from, match_to):
    """
    This function deletes files from delete_from that do not match the files in match_to.
    """
    # Get the list of files in the directory
    files_to_delete = os.listdir(delete_from)
    files_to_match = os.listdir(match_to)

    # Iterate over each file in the directory
    for filename in files_to_delete:
        if filename not in files_to_match:
            filepath = os.path.join(delete_from, filename)
            os.remove(filepath)
            
def delete_non_matching_labels(delete_from_txt, match_to):
    """
    This function deletes files from delete_from that do not match the files in match_to.
    """
    # Get the list of files in the directory
    files_to_delete = os.listdir(delete_from_txt)
    files_to_match = os.listdir(match_to)

    # Iterate over each file in the directory
    for filename in files_to_delete:
        if filename.rsplit('.', 1)[0] + '.jpg' not in files_to_match:
            filepath = os.path.join(delete_from_txt, filename)
            os.remove(filepath)
            

delete_non_matching_jpg(delete_from, match_to)
delete_non_matching_labels(delete_from_txt, match_to)
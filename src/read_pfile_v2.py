import numpy as np
import sys
import read_data

def read_pfile(pfile_name, read_only_frame_table=False, sent_indices=(None, None), verbose = False):
    sent_start = sent_indices[0]
    sent_end = sent_indices[1]
    fpi = open(pfile_name)
    header_line = fpi.readline().strip()
    ref_header = '-pfile_header version 0 size 32768'
    if header_line != ref_header:
        raise ValueError('the header suggests that this is not a pfile')

    line = fpi.readline().strip().split(" ")
    while line[0] != '-end':
        if line[0] == '-num_sentences':
            num_sents = int(line[1])
        elif line[0] == '-num_frames':
            num_frames = int(line[1])
        elif line[0] == '-first_feature_column':
            first_feature_column = int(line[1])
        elif line[0] == '-num_features':
            num_feats = int(line[1])
        elif line[0] == '-first_label_column':
            first_label_column = int(line[1])
        elif line[0] == '-num_labels':
            num_labels = int(line[1])
        elif line[0] == '-format':
            format_self = line[1]
        elif " ".join(line[0:2]) == '-data size':
            data_size = int(line[2])
        elif line[0] == '-sent_table_data':
            extra_info_table_data = line[1:]
        else:
            raise IOError("".join(["Illegal parameter", line, "probably corrupted pfile"]))#raise IOError    
        line = fpi.readline().strip().split(" ")

    #fpi.close()
    #fpi = open(pfile_name, 'rb')
    frame_size = (num_feats + num_labels + 2) * 4
    if sent_start != None and sent_end != None:
        pass
    else:
        sent_start = 0
        sent_end = num_sents
    num_read_sents = sent_end - sent_start
    fpi.seek(32768 + frame_size * num_frames + sent_start * 4)
    cum_sum_frame_table = np.fromfile(fpi, dtype=">I", count=num_read_sents+1)
    start_frame = cum_sum_frame_table[0]
    end_frame = cum_sum_frame_table[-1]
    num_read_frames = end_frame - start_frame

    if read_only_frame_table:
        cum_sum_frame_table -= cum_sum_frame_table[0]
        return num_feats, cum_sum_frame_table
    
    feature_dtype = "".join([">", str(num_feats), "f"])
    label_dtype = "".join([">", str(num_labels), "I"])
    record_dtype = [('frames', ">2I")]
    if first_feature_column < first_label_column:
        record_dtype.append(('features', feature_dtype))
        record_dtype.append(('labels', label_dtype))
    else:
        record_dtype.append(('labels', label_dtype))
        record_dtype.append(('features', feature_dtype))
    record_dtype = np.dtype(record_dtype) 
    fpi.seek(32768 + frame_size * start_frame)
    output_data = np.fromfile(fpi, dtype=record_dtype, count = num_read_frames)
    output_data['frames'][:,0] -= output_data['frames'][0,0]
    cum_sum_frame_table -= cum_sum_frame_table[0]
    
    
#    readed_data = np.empty((num_read_frames, num_feats), order='C', dtype=np.float32)
#    read_data.read_feature_data(fpi.fileno(), readed_data, 32768 + frame_size * start_frame, num_labels)
#    readed_data.byteswap(True)
#    print "C-data"
#    print readed_data
    
    fpi.close()
    
    new_data = np.empty(output_data['features'].shape, order='C', dtype=float)
    new_data[:] = output_data['features'][:]
    new_labels = np.empty(output_data['labels'].shape, order='C', dtype = int)
    new_labels[:] = output_data['labels'][:]
#    print "python data"
#    print new_data
    
    
    
#    quit()
    
    return new_data, output_data['frames'], new_labels, cum_sum_frame_table

def frame_table_to_frames(frame_table):
    num_examples = frame_table[-1] - frame_table[0]
    frames = np.zeros((num_examples, 2), dtype=int)
    current_frame = 0
    for next_sent_index in range(1,len(frame_table)):
        sent_index = next_sent_index - 1
        num_frames = frame_table[next_sent_index] - frame_table[sent_index]
        end_frame = current_frame + num_frames
        frames[current_frame:end_frame,0] = sent_index
        frames[current_frame:end_frame,1] = range(num_frames)
        current_frame = end_frame
    return frames
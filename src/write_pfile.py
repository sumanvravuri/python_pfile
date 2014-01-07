import numpy as np
import struct
import sys

def write_pfile(pfile_name, frames, data = None, labels = None):
    if data == None and labels == None:
        raise ValueError('Either data or labels or both need to be used')
    if data == None:
        num_feats = 0
    else:
	num_feats = data.shape[1]
    if labels == None:
        num_labels = 0
    else:
	num_labels = labels.shape[1]

    write_pfile_header(pfile_name, frames, num_feats, num_labels)
    write_pfile_data(pfile_name, frames, data, labels)
    write_pfile_frame_table(pfile_name, frames)

def write_pfile_header(pfile_name, frames, num_feats, num_labels):
    fpi = open(pfile_name, 'w')
    num_sentences = frames[-1,0] + 1
    num_frames = frames.shape[0]
    first_feat_column = frames.shape[1]

    first_label_column = first_feat_column + num_feats

    format = 'd' * first_feat_column + 'f' * num_feats + 'd' * num_labels

    num_cols = num_feats + num_labels + first_feat_column
    data_size = num_cols * num_frames
    first_label_column = first_feat_column + num_feats
    fpi.write('-pfile_header version 0 size 32768\n')
    fpi.write('-num_sentences %d\n' % num_sentences)
    fpi.write('-num_frames %d\n' % num_frames)
    fpi.write('-first_feature_column %d\n' % first_feat_column)
    fpi.write('-num_features %d\n' % num_feats)
    fpi.write('-first_label_column %d\n' % first_label_column)
    fpi.write('-num_labels %d\n' % num_labels)
    fpi.write('-format %s\n' % format)
    fpi.write('-data size %d offset 0 ndim %d nrow %d ncol %d\n' % (data_size, first_feat_column, num_frames, num_cols))
    fpi.write('-end\n')
    current_position = fpi.tell()
    fpi.write("".join([struct.pack('>B', 0)] * (32768 - current_position)))

    fpi.close()

def write_pfile_data(pfile_name, frames, data = None, labels = None):
    fpi = open(pfile_name, 'a')
    num_sentences = frames[-1,0] - frames[0,0] + 1
    num_frames = frames.shape[0]
    first_feat_column = frames.shape[1]
    if data is None:
        num_feats = 0
    else:
	num_feats = data.shape[1]

    first_label_column = first_feat_column + num_feats
    if labels is None:
        num_labels = 0
    else:
	num_labels = labels.shape[1]

    format_for_file = 'I' * first_feat_column + 'f' * num_feats + 'I' * num_labels
    
    num_cols = num_feats + num_labels + first_feat_column
    data_size = num_cols * num_frames
    frame_byte_size = len(format_for_file) * 4
    num_frames_per_chunk = max((2**10) / frame_byte_size, 1)
    for start_index in range(0,num_frames, num_frames_per_chunk):
    	end_index = min(start_index+num_frames_per_chunk, num_frames)
	num_frame_per_chunk = end_index - start_index
	buffer = np.empty((num_frame_per_chunk, num_cols))
	buffer[:,:first_feat_column] = frames[start_index:end_index, :]
	if num_feats > 0:
	    buffer[:,first_feat_column:first_feat_column+num_feats] = data[start_index:end_index, :]
	if num_labels > 0:
    	    buffer[:,first_label_column:first_label_column+num_labels] = labels[start_index:end_index, :]
	
	buffer = buffer.reshape((buffer.size,)).tolist()
	output_format = '>' + format_for_file * num_frame_per_chunk
	fpi.write(struct.pack(output_format, *buffer))

    fpi.close()

def write_pfile_frame_table(pfile_name, frames):
    fpi = open(pfile_name, 'a')
    last_frame_dim = frames.shape[1] - 1
    frame_table = np.hstack((np.where(frames[:,last_frame_dim] == 0)[0], frames.shape[0])).tolist()
    num_items = len(frame_table)
    fpi.write(struct.pack('>' + num_items * 'I', *frame_table))
    fpi.close()

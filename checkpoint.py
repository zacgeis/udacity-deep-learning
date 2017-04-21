# Jupyter Notebook is currently not saving.

# Encoding
lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers)
initial_state = state = stacked_lstm.zero_state(rnn_inputs.get_shape()[0], tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=initial_state)
rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
return initial_state

# Decoding - Training
decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
outputs, final_state, final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, decoder_fn, inputs=dec_embed_input, sequence_length=sequence_length, scope=decoding_scope)
outputs = tf.nn.dropout(outputs, keep_prob)
return output_fn(outputs)

# Decoding - Inference
decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size)
outputs, final_state, final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, decoder_fn, scope=decoding_scope)
outputs = tf.nn.dropout(outputs, keep_prob)
return outputs

# decoding layer
with tf.variable_scope('decoder') as scope:
	lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
	stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers)
	initial_state = state = stacked_lstm.zero_state(dec_embed_input.get_shape()[0], tf.float32)
	rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, dec_embed_input, initial_state=initial_state)
	rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

	output_fn = lambda outputs: tf.contrib.layers.linear(outputs, vocab_size, scope=scope)
	training_logits = decoding_layer_train(encoder_state, stacked_lstm, dec_embed_input, sequence_length, scope, output_fn, keep_prob)

	scope.reuse_variables()
	start_of_sequence_id = target_vocab_to_int['<GO>']
	end_of_sequence_id = target_vocab_to_int['<EOS>']
	maximum_length = sequence_length
	inference_logits = decoding_layer_infer(encoder_state, stacked_lstm, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, scope, output_fn, keep_prob)

return (training_logits, inference_logits)

# build the network


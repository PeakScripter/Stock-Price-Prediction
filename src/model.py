from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Bidirectional, Dropout, Input, Flatten, Concatenate, Permute, Reshape, Multiply
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def build_lstm_cnn_model(input_shape, config):
    """Bulid LSTM-CNN model."""
    model = Sequential()
    model.add(Conv1D(filters=config['filters_cnn'], kernel_size=config['kernel_size'], activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=config['pool_size']))
    model.add(LSTM(units=config['units_lstm'], return_sequences=False))
    model.add(Dropout(config['dropout']))
    model.add(Dense(config['units_dense'], activation='relu'))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_bidirectional_lstm_model(input_shape, config):
    """Build Bidirectional LSTM model."""
    model = Sequential()
    model.add(Bidirectional(LSTM(units=config['units_lstm'], return_sequences=True), input_shape=input_shape))
    model.add(Dropout(config['dropout']))
    model.add(Bidirectional(LSTM(units=config['units_lstm'], return_sequences=False)))
    model.add(Dropout(config['dropout']))
    model.add(Dense(config['units_dense'], activation='relu'))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def attention_block(inputs, time_step):
    """
    Attention layer implementation.
    """
    # inputs.shape = (batch_size, time_steps, input_dim)
    # Permute to (batch_size, input_dim, time_steps) not explicitly needed if we handle dims right,
    # but standard attention often calculates weights per time step.
    
    # Calculate attention scores
    # Dense layer to compute score for each step
    a = Permute((2, 1))(inputs) # (batch, input_dim, time_steps)
    a = Reshape((inputs.shape[2], inputs.shape[1]))(a) 
    a = Dense(inputs.shape[1], activation='softmax')(a) # (batch, input_dim, time_steps) - softmax over time steps
    
    a_probs = Permute((2, 1), name='attention_vec')(a) # (batch, time_steps, input_dim)
    
    # Apply weights
    output_attention_mul = Multiply()([inputs, a_probs]) # (batch, time_steps, input_dim)
    return output_attention_mul

def build_attention_lstm_model(input_shape, config):
    """Build LSTM model with Attention."""
    inputs = Input(shape=input_shape)
    
    # LSTM layer
    lstm_out = LSTM(config['units_lstm'], return_sequences=True)(inputs)
    lstm_out = Dropout(config['dropout'])(lstm_out)
    
    # Attention
    # Note: Traditional attention allows focusing on different parts of the sequence.
    # Here we define a simple self-attention mechanism to weight the LSTM outputs.
    
    # Simple Attention using a Dense layer to learn importance of each time step
    # We want a single vector representing the sequence context.
    
    # Flatten the LSTM output (batch, time_steps, units) -> need to weigh it.
    # Let's try a standard attention mechanism often used in time series.
    
    # Attention:
    # 1. Score: e = tanh(W * h + b)
    # 2. Alpha: a = softmax(e)
    # 3. Context: c = sum(a * h)
    
    # Keras functional implementation:
    # Attention weights
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = Dense(input_shape[0], activation='softmax')(attention) # Softmax over time steps? No, time_steps is dimension 1
    # Actually, input_shape[0] is time_steps.
    attention = Dense(input_shape[0], activation='softmax', name='attention_vec')(attention) # (batch, time_steps)
    attention = Reshape((input_shape[0], 1))(attention) # (batch, time_steps, 1) to broadcast
    
    attention_mul = Multiply()([lstm_out, attention]) # Weight the LSTM output states
    
    # Sum over time steps or flatten/pool
    # Context vector is usually the sum
    # But here we can also just flatten or use another LSTM/Dense
    
    # Let's Flatten
    flatten = Flatten()(attention_mul)
    
    output = Dense(config['units_dense'], activation='relu')(flatten)
    output = Dense(1)(output)
    
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def create_model(input_shape, config):
    """Factory function to create model based on config."""
    model_type = config.get('type', 'lstm_cnn')
    
    if model_type == 'lstm_cnn':
        return build_lstm_cnn_model(input_shape, config)
    elif model_type == 'bidirectional_lstm':
        return build_bidirectional_lstm_model(input_shape, config)
    elif model_type == 'attention_lstm':
        return build_attention_lstm_model(input_shape, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

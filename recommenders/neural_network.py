import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from products.models import ProductRating


def create_neural_network(EMBEDDING_SIZE, NUM_MOVIES, NUM_USERS, ROW_COUNT):
    movie_input = keras.Input(shape=(1,), name='movie_id')

    movie_emb = layers.Embedding(output_dim=EMBEDDING_SIZE, input_dim=NUM_MOVIES, input_length=ROW_COUNT,
                                 name='movie_emb')(movie_input)
    movie_vec = layers.Flatten(name='FlattenMovie')(movie_emb)

    movie_model = keras.Model(inputs=movie_input, outputs=movie_vec)

    user_input = keras.Input(shape=(1,), name='user_id')

    user_emb = layers.Embedding(output_dim=EMBEDDING_SIZE, input_dim=NUM_USERS, input_length=ROW_COUNT,
                                name='user_emb')(user_input)
    user_vec = layers.Flatten(name='FlattenUser')(user_emb)

    user_model = keras.Model(inputs=user_input, outputs=user_vec)

    merged = layers.Dot(name='dot_product', normalize=True, axes=2)([movie_emb, user_emb])
    merged_dropout = layers.Dropout(0.2)(merged)

    dense_1 = layers.Dense(70, name='FullyConnected-1')(merged_dropout)
    dropout_1 = layers.Dropout(0.2, name='Dropout_1')(dense_1)

    dense_2 = layers.Dense(50, name='FullyConnected-2')(dropout_1)
    dropout_2 = layers.Dropout(0.2, name='Dropout_2')(dense_2)

    dense_3 = keras.layers.Dense(20, name='FullyConnected-3')(dropout_2)
    dropout_3 = keras.layers.Dropout(0.2, name='Dropout_3')(dense_3)

    dense_4 = keras.layers.Dense(10, name='FullyConnected-4', activation='relu')(dropout_3)

    result = layers.Dense(1, name='result', activation="relu")(dense_4)

    adam = keras.optimizers.Adam(lr=0.001)
    model = keras.Model([movie_input, user_input], result)
    model.compile(optimizer=adam, loss='mean_absolute_error')
    return model, movie_model, user_model


def train_model():
    ratings = list(ProductRating.objects.values_list('user_id', 'product_id', 'rating'))
    ratings = np.asarray(ratings)
    train, test = train_test_split(ratings, test_size=0.1)

    movie_ids = ProductRating.objects.values_list('product_id', flat=True).distinct()
    num_movies = movie_ids.count()

    user_ids = ProductRating.objects.values_list('user_id', flat=True).distinct()
    num_users = user_ids.count()

    EMBEDDING_SIZE = 10
    row_count = train.shape[0]

    model, movie_model, user_model = create_neural_network(EMBEDDING_SIZE, num_movies+1, num_users+1, row_count)

    callbacks = [keras.callbacks.EarlyStopping('val_loss', patience=10),
                 keras.callbacks.ModelCheckpoint('besttest.h5', save_best_only=True)]

    train_user_ids = np.asarray([obj[0] for obj in train])
    train_movie_ids = np.asarray([obj[1] for obj in train])
    train_ratings = np.asarray([obj[2] for obj in train])

    test_user_ids = np.asarray([obj[0] for obj in test])
    test_movie_ids = np.asarray([obj[1] for obj in test])
    test_ratings = np.asarray([obj[2] for obj in test])

    history = model.fit(
        [train_movie_ids, train_user_ids],
        train_ratings,
        batch_size=100,
        epochs=10,
        validation_data=([test_movie_ids, test_user_ids], test_ratings),
        verbose=2,
        callbacks=callbacks
    )

    model.save('recommenders/saved_models/main_model.h5')
    user_model.save('recommenders/saved_models/user_model.h5')
    movie_model.save('recommenders/saved_models/movie_model.h5')

    #### Save product embedding to pickle file ####
    product_embedding_list = []

    for _id in movie_ids:
        emb = movie_model.predict(np.array([_id]))
        val = list(emb.reshape(1, -1))[0]
        product_embedding_list.insert(_id, val)

    with open('recommenders/saved_models/product_embedding_list.pkl', 'wb') as f:
        pickle.dump(product_embedding_list, f)


def load_model():
    main_model = models.load_model('recommenders/saved_models/main_model.h5')
    user_model = models.load_model('recommenders/saved_models/user_model.h5')
    movie_model = models.load_model('recommenders/saved_models/movie_model.h5')

    return main_model, user_model, movie_model


def get_recommendations_for_user(user_id):
    main_model, user_model, movie_model = load_model()

    # If user_id does not exist in embedding, return empty list
    try:
        user_embedding = user_model.predict([user_id]).reshape(1, -1)[0]
    except InvalidArgumentError:
        return []

    unique_product_ids = ProductRating.objects.values_list('product_id', flat=True).distinct()

    with open('recommenders/saved_models/product_embedding_list.pkl', 'rb') as f:
        product_embedding_list = pickle.load(f)

    knn_train_label = unique_product_ids

    clf = KNeighborsClassifier(n_neighbors=11)
    clf.fit(product_embedding_list, knn_train_label)

    distances, indices = clf.kneighbors(user_embedding.reshape(1, -1), n_neighbors=10)
    recommended_product_ids = indices.reshape(10, 1)
    recommended_product_ids = [item[0] for item in list(recommended_product_ids)]

    return recommended_product_ids

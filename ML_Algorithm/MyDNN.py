import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class MyDNN:
    def __init__(self):
        pass

    @classmethod
    def _run_dnn(cls, X_train, y_train, X_test, y_test, fig_save_path, stat_cycle_data_df, stat_cycle_label_df) \
            -> tuple:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        nadam_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(optimizer=nadam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100,
                                             restore_best_weights=True, mode='max')
        ]

        history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                            epochs=1000, batch_size=32, callbacks=callbacks)

        y_pred = model.predict(X_test_scaled)
        y_pred = (y_pred > 0.5).astype(int)

        plt.figure(figsize=(16, 8))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(fig_save_path)
        plt.close()

        result = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        stat_cycle_scaled = scaler.transform(stat_cycle_data_df)
        stat_cycle_pred = model.predict(stat_cycle_scaled)
        stat_cycle_pred = (stat_cycle_pred > 0.5).astype(int)

        stat_cycle_result \
            = classification_report(stat_cycle_label_df, stat_cycle_pred, zero_division=0, output_dict=True)

        return result, stat_cycle_result

import tensorflow as tf
from tensorflow.keras.layers import(
    Input, Dense, Dropout, Flatten
)
from tensorflow.keras.models import Model   
from tensorflow.keras.applications import VGG16, VGG19, Xception, InceptionV3, ResNet50, MobileNetV2
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
import pandas as pd


class Models():
    def __init__(self, model_name, img_height, img_width, fine_tuning=False, output_layers='default'):
        """
        Initializes the Models class with a specific model name and parameters.
        
        :param model_name: Name of the model to be instantiated (e.g., 'VGG16', 'Xception').
        :param img_height: Height of the input images.
        :param img_width: Width of the input images.
        :param fine_tuning: Whether to enable fine-tuning for transfer learning.
        :param output_layers: Variant of the model output (e.g., 'simple', 'branch', 'default').
        """
        self.model_name = model_name
        self.img_height = img_height
        self.img_width = img_width
        self.fine_tuning = fine_tuning
        self.output_layers = output_layers
    
    def get_model(self):
        """
        Returns the compiled model based on the specified parameters.
        
        :return: Compiled Keras model.
        """
        return self.transfer_learning(
            height=self.img_height, 
            width=self.img_width, 
            model=self.model_name, 
            fine_tuning=self.fine_tuning, 
            output_layers=self.output_layers
        )

    @staticmethod
    def transfer_learning(height, width, model="VGG16", fine_tuning=False, output_layers="default"):
        """
        Applies transfer learning using pre-trained models.
        
        :param height: Height of the input images.
        :param width: Width of the input images.
        :param model: Name of the pre-trained model (e.g., 'VGG16', 'InceptionV3').
        :param fine_tuning: Whether to enable fine-tuning.
        :param output_layers: Variant of the model output.
        :return: Compiled Keras model.
        """
        
        input_img = Input(shape=(height, width, 3), name='input', dtype=tf.float32)

        # Select the appropriate base model
        base_model = Models.base_model(model, input_img)
        
        # Fine-tuning setup
        if not fine_tuning:
            base_model.trainable = False
        else:    
            base_model.trainable = True
            set_trainable = False
            for layer in base_model.layers:
                if layer.name in ['block5_conv1', 'batch_normalization']:  # You can add more blocks here
                    set_trainable = True
                layer.trainable = set_trainable
                
        # Model construction
        x = base_model.output
        x = Flatten()(x)
        
        output1, output2 = Models.model_variant(x, output_layers)
        
        model = Model(inputs=input_img, outputs=[output1, output2])
        
        # Display the trainable status of layers
        layers_info = [(layer, layer.name, layer.trainable) for layer in base_model.layers]
        print(pd.DataFrame(layers_info, columns=['Layer Type', 'Layer Name', 'Layer Trainable']).head(1000))

        return model
    
    @staticmethod
    def base_model(model_name, input_tensor):
        """
        Selects and initializes a pre-trained model based on the model name.
        
        :param model_name: Name of the model to be initialized.
        :param input_tensor: Input tensor for the model.
        :return: Initialized pre-trained model.
        """
        if model_name == "VGG16":
            return VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif model_name == "VGG19":
            return VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif model_name == "Xception":
            return Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif model_name == "InceptionV3":
            return InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif model_name == "ResNet50":
            return ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        elif model_name == "MobileNetV2":
            return MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
    
    @staticmethod
    def model_variant(x, variant="default"):
        """
        Configures the model variant with different output layer configurations.
        
        :param x: Input tensor.
        :param variant: Variant type (e.g., 'simple', 'branch', 'default').
        :return: Output layers for the model.
        """
        if variant == "simple":
            # Simple Dense Layers
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.25)(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.25)(x)
            output1 = Dense(40, activation='softmax', name='Class')(x)
            output2 = Dense(1, activation='sigmoid', name='MoreThanOnePerson')(x)
        
        elif variant == 'branch':
            # Branching Layers
            # Branch 1
            x1 = Dense(512, activation='relu')(x)
            x1 = Dropout(0.5)(x1)
            x1 = Dense(512, activation='relu')(x1)
            x1 = Dropout(0.5)(x1)
            output1 = Dense(40, activation='softmax', name='Class')(x1)
            # Branch 2
            x2 = Dense(512, activation='relu')(x)
            x2 = Dropout(0.5)(x2)
            x2 = Dense(512, activation='relu')(x2)
            x2 = Dropout(0.5)(x2)
            output2 = Dense(1, activation='sigmoid', name='MoreThanOnePerson')(x2)
        
        else:
            # Default Common Branch
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.25)(x)
            # Branch 1    
            x1 = Dense(512, activation='relu')(x)
            x1 = Dropout(0.25)(x1)
            output1 = Dense(40, activation='softmax', name='Class')(x1)
            # Branch 2
            x2 = Dense(512, activation='relu')(x)
            x2 = Dropout(0.25)(x2)
            output2 = Dense(1, activation='sigmoid', name='MoreThanOnePerson')(x2)
            
        return output1, output2



# ===============================
# Basic Exploratory Data Analysis
# ===============================

class EDA:

    @staticmethod
    def plot_distribution(pd_series):
        labels = pd_series.value_counts().index.tolist()
        counts = pd_series.value_counts().values.tolist()

        pie_plot = go.Pie(labels=labels, values=counts, hole=.3)
        fig = go.Figure(data=[pie_plot])
        fig.update_layout(title_text='Distribution for %s' % pd_series.name)

        fig.show()

    @staticmethod
    def plot_class_distribution(df):
        fig = px.histogram(df, x="Class", nbins=20)
        fig.update_layout(title_text='Class distribution')
        fig.show()

    @staticmethod
    def plot_targets(df):
        classes = df['Class'].value_counts()
        persons = df['MoreThanOnePerson'].value_counts()

        fig, axs = plt.subplots(1, 2, figsize=(17, 8), sharey=False)
        fig.suptitle('Target Categorical Plotting', weight='bold', fontsize=16)

        bars1 = axs[0].barh(classes.index, classes.values, color='#0C457D')
        axs[0].set_xlabel('', weight='bold', fontsize=12)
        axs[0].tick_params(axis='x', labelsize=10)
        axs[0].tick_params(axis='y', labelsize=10,)

        for bar in bars1:
            w = bar.get_width()
            axs[0].text(w, bar.get_y() + bar.get_height() / 2, str(int(w)), ha='left', va='center', fontsize=12)

        bars2 = axs[1].barh(persons.index, persons.values, color='#0C457D')
        axs[1].set_ylabel('')
        axs[1].tick_params(axis='x', labelsize=10)
        axs[1].tick_params(axis='y', labelsize=10)

        for bar in bars2:
            w = bar.get_width()
            axs[1].text(w, bar.get_y() + bar.get_height() / 2, str(int(w)), ha='left', va='center', fontsize=12)

        plt.tight_layout()
        plt.show()
        
        
    @staticmethod
    def plot_highlevelcategory_relationship(df):
        # Plot relationship between HighLevelCategory and Class
        fig = px.histogram(df, x='Class', color='HighLevelCategory', 
                           category_orders={'HighLevelCategory': sorted(df['HighLevelCategory'].unique())},
                           title='Distribution of Class by HighLevelCategory')
        fig.update_layout(barmode='stack')
        fig.show()

        # Plot relationship between HighLevelCategory and MoreThanOnePerson
        fig = px.histogram(df, x='MoreThanOnePerson', color='HighLevelCategory', 
                           category_orders={'HighLevelCategory': sorted(df['HighLevelCategory'].unique())},
                           title='Distribution of MoreThanOnePerson by HighLevelCategory')
        fig.update_layout(barmode='stack')
        fig.show()
        
    @staticmethod
    def plot_contingency_heatmaps(df):
        cross_tab_class_highlevel = pd.crosstab(df['Class'], df['HighLevelCategory'])
        cross_tab_morethanone_highlevel = pd.crosstab(df['MoreThanOnePerson'], df['HighLevelCategory'])
        
        chi2_class, p_class, _, _ = chi2_contingency(cross_tab_class_highlevel)
        chi2_morethanone, p_morethanone, _, _ = chi2_contingency(cross_tab_morethanone_highlevel)

        fig, axs = plt.subplots(1, 2, figsize=(18, 12), sharey=False)
        fig.suptitle('Contingency Tables Heatmap', weight='bold', fontsize=16)

        sns.heatmap(cross_tab_class_highlevel, annot=True, fmt='d', cmap='Blues', ax=axs[0], cbar=False, linewidths=0.5, linecolor='#0C457D')
        axs[0].set_title(f'Class vs HighLevelCategory\nChi2: {chi2_class:.2f}, p-value: {p_class:.4f}')
        axs[0].tick_params(axis='x', labelsize=10)
        axs[0].tick_params(axis='y', labelsize=10)

        sns.heatmap(cross_tab_morethanone_highlevel, annot=True, fmt='d', cmap='Oranges', ax=axs[1], cbar=False, linewidths=0.5, linecolor='#0C457D')
        axs[1].set_title(f'MoreThanOnePerson vs HighLevelCategory\nChi2: {chi2_morethanone:.2f}, p-value: {p_morethanone:.4f}')
        axs[1].tick_params(axis='x', labelsize=10)
        axs[1].tick_params(axis='y', labelsize=10)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
            
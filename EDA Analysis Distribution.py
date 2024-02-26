import os
import matplotlib.pyplot as plt


def plot_emotion_distribution(base_path, output_graph_path):
    # Define emotions
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Initialize a dictionary to hold the count of images for each emotion
    emotion_counts = {emotion: 0 for emotion in emotions}

    # Count the number of images in each emotion's directory
    for emotion in emotions:
        emotion_dir = os.path.join(base_path, emotion)
        try:
            emotion_counts[emotion] = len(
                [name for name in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, name))])
        except FileNotFoundError:
            print(f"Directory not found for {emotion}, setting count to 0.")
            emotion_counts[emotion] = 0

    # Plotting the distribution of images across emotions
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images by Emotion')
    plt.xticks(rotation=45)

    # Save the plot
    plt.savefig(output_graph_path)
    plt.close()


# Example usage
base_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Work-data/'
output_graph_path = '/home/orion/Geo/Projects/FREA-Facial-Recognition-and-Emotion-Analysis/Emotion_Distribution.png'
plot_emotion_distribution(base_path, output_graph_path)

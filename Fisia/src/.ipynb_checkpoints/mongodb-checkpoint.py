from pymongo import MongoClient


def fetch_correct_exercises(exercise_name, professional_id,
                            db_uri='mongodb://root:root@192.168.219.38:27017/?authMechanism=DEFAULT',
                            db_name='fisia', collection_name='videos', correct_label=1):
    """
    Fetch all annotations from a professional on a specific exercise from MongoDB where the execution may be correct or incorrect.

    Parameters:
        exercise_name (str): The title of the exercise to query.
        professional_id (int): The ID of the professional whose annotations are to be fetched.
        db_uri (str): URI for connecting to MongoDB.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection where exercise data is stored.
        correct_label (int, optional): The label indicating whether the exercise is performed correctly. Default is 1.

    Returns:
        list: A list of documents containing the exercise data matching the criteria.
    """
    # Connect to MongoDB
    client = MongoClient(db_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Criteria for matching annotations in the array
    annotations_criteria = {
        'professional_id': professional_id,
        'correct_label': correct_label
    }

    # Execute the query
    query = {
        'program_item_title': exercise_name,
        'annotations': {'$elemMatch': annotations_criteria}
    }
    results = collection.find(query)

    # Convert cursor to list and return
    return list(results)


def extract_annotations(document):
    video_api = document['activity_video_path_api']
    print(video_api)
    annotations_data = []
    for annotations in document.get('annotations', []):
        annot = {
            'start': annotations['start'],
            'end': annotations['end'],
            'reps': annotations['reps'],
            'position': annotations['pos_patient'],
            'correct_label_range': annotations['correct_label_range']
        }
        annotations_data.append(annot)
    return annotations_data


def extract_keypoints(document):
    """
    Extracts keypoints data from each document contains multiple keypoints for frames.

    Parameters:
        document:  document is expected to have a 'keypoints' list.

    Returns:
        list of list of dict: A nested list where each inner list represents keypoints data for one frame,
                              including keypoint name, x and y coordinates, and score.
    """
    all_frames_keypoints = []

    # Loop through each set of keypoints in the document
    for keypoints_data in document.get('keypoints', []):
        frame_data = [
            {
                'name': keypoint['name'],
                'x': keypoint['x'],
                'y': keypoint['y'],
                'score': keypoint['score']
            }
            for keypoint in keypoints_data.get('keypoints', [])
        ]
        all_frames_keypoints.append(frame_data)
    return all_frames_keypoints
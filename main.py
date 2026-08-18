import cv2
import face_recognition
import json
from pathlib import Path
from datetime import datetime


# =====================================================
# SETTINGS
# =====================================================

KNOWN_FOLDER = "known_faces"
INPUT_IMAGE = "input/test.jpg"
OUTPUT_FOLDER = "output/annotated_outputs"
LOG_FILE = "logs/recognition_log.json"

# Smaller value = stricter matching
THRESHOLD = 0.60


# =====================================================
# LOAD KNOWN FACES
# =====================================================

def load_known_faces():

    known_encodings = []
    known_names = []

    known_path = Path(KNOWN_FOLDER)

    if not known_path.exists():
        print("ERROR: known_faces folder not found.")
        return known_encodings, known_names

    for person_folder in known_path.iterdir():

        if not person_folder.is_dir():
            continue

        person_name = person_folder.name

        for image_file in person_folder.iterdir():

            if image_file.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:
                continue

            print("Loading:", image_file)

            image = face_recognition.load_image_file(
                str(image_file)
            )

            locations = face_recognition.face_locations(
                image
            )

            if len(locations) != 1:

                print(
                    "Skipping:",
                    image_file,
                    "- exactly one face required."
                )

                continue

            encodings = face_recognition.face_encodings(
                image,
                locations
            )

            if len(encodings) == 1:

                known_encodings.append(encodings[0])
                known_names.append(person_name)

                print(
                    "Known person loaded:",
                    person_name
                )

    return known_encodings, known_names


# =====================================================
# RECOGNIZE FACES
# =====================================================

def recognize_faces(
    image,
    known_encodings,
    known_names
):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    face_locations = face_recognition.face_locations(
        rgb_image
    )

    face_encodings = face_recognition.face_encodings(
        rgb_image,
        face_locations
    )

    results = []

    for location, face_encoding in zip(
        face_locations,
        face_encodings
    ):

        # If there are no known faces
        if len(known_encodings) == 0:

            name = "Unknown"
            distance = 1.0

        else:

            distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_index = distances.argmin()
            distance = float(distances[best_index])

            if distance <= THRESHOLD:

                name = known_names[best_index]

            else:

                name = "Unknown"

        # This is a simple similarity score,
        # not a calibrated probability.
        confidence = max(
            0.0,
            min(
                1.0,
                1.0 - distance
            )
        )

        top, right, bottom, left = location

        results.append({

            "identity": name,

            "confidence": round(
                confidence,
                4
            ),

            "distance": round(
                distance,
                4
            ),

            "bounding_box": {

                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left
            }
        })

    return results


# =====================================================
# ANNOTATE IMAGE
# =====================================================

def annotate_image(
    image,
    results
):

    for result in results:

        top = result["bounding_box"]["top"]
        right = result["bounding_box"]["right"]
        bottom = result["bounding_box"]["bottom"]
        left = result["bounding_box"]["left"]

        name = result["identity"]
        confidence = result["confidence"]

        # Draw bounding box
        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Label
        label = (
            f"{name} | "
            f"Confidence: {confidence:.2f}"
        )

        cv2.putText(
            image,
            label,
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return image


# =====================================================
# SAVE JSON LOG
# =====================================================

def save_log(
    source,
    results
):

    log_path = Path(LOG_FILE)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {

        "timestamp":
            datetime.now().isoformat(),

        "source":
            source,

        "faces":
            results
    }

    existing_logs = []

    if log_path.exists():

        try:

            with open(
                log_path,
                "r",
                encoding="utf-8"
            ) as file:

                existing_logs = json.load(file)

        except:

            existing_logs = []

    existing_logs.append(record)

    with open(
        log_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            existing_logs,
            file,
            indent=4
        )


# =====================================================
# PROCESS IMAGE
# =====================================================

def process_image():

    print()
    print("==============================")
    print("IMAGE PROCESSING")
    print("==============================")

    image = cv2.imread(
        INPUT_IMAGE
    )

    if image is None:

        print(
            "ERROR: input/test.jpg not found."
        )

        return

    print(
        "Image loaded successfully."
    )

    results = recognize_faces(
        image,
        known_encodings,
        known_names
    )

    print(
        "Faces detected:",
        len(results)
    )

    for result in results:

        print("------------------------------")
        print(
            "Identity:",
            result["identity"]
        )

        print(
            "Distance:",
            result["distance"]
        )

        print(
            "Confidence:",
            result["confidence"]
        )

    image = annotate_image(
        image,
        results
    )

    output_path = Path(
        OUTPUT_FOLDER
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_path /
        "annotated_result.jpg"
    )

    cv2.imwrite(
        str(output_file),
        image
    )

    save_log(
        INPUT_IMAGE,
        results
    )

    print()
    print("SUCCESS!")
    print(
        "Annotated image:",
        output_file
    )

    print(
        "JSON log:",
        LOG_FILE
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    known_encodings, known_names = (
        load_known_faces()
    )

    print()
    print(
        "Total known faces:",
        len(known_encodings)
    )

    process_image()
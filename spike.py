import partitura as pt

score_path = "/Users/jiyun/workspace/asap-dataset/Bach/Fugue/bwv_858/xml_score.musicxml"


if __name__ == "__main__":
    # Load the score
    xml_score = pt.load_musicxml(score_path)

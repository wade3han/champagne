"""
Define the prompt for different tasks.
"""
Prompt_prefix_image_text = [
    'An image of '
]
Prompt_Image_Tagging = [
    'What is this in the image ?'
]
Prompt_Image_Tagging_Scene = [
    'What is the category of the scene ?'
]
Prompt_Image_Generation = [
    'What is the complete image? Text: " {} " .'
]
Prompt_Object_Detection = [
    'Which region does the text " {} " describe ? '
]
Prompt_Box_Classification_Scene = [
    'What is the category of region " {} " ?'
]
Prompt_Object_Detection_All_Class = [
    'Locate all objects in the image .'
]
Prompt_Object_Segmentation = [
    'What is the segmentation of " {} " ?',
]
Prompt_Depth_Estimation = [
    'What is the depth map of the image ?'
]
Prompt_Surface_Normals_Estimation = [
    'What is the surface normal of the image ?'
]
Prompt_Image_Inpainting = [
    'Filling the blank region " {} " ?'
]
Prompt_Pose_Estimation = [
    'Find the human joints in the region " {} " .'
]
Prompt_Refer_Expression = [
    'Which region does the text " {} " describe ?'
]
Prompt_Image_Captioning = [
    'What does the image describe ?'
]
Prompt_Image_Localized_Narrative = [
    'Describe the image with narratives .'
]
Prompt_Region_Captioning = [
    'What does the region " {} " describe ?'
]
Prompt_Visual_Entailment = [
    'Can image and text " {1} " imply text " {2} " ?'
]
Prompt_Relationship_Tagging = [
    'What is the relationship between " {1} " and " {2} " ?'
]
Prompt_VCR_QA = [
    'Question: " {1} " Answer: '
]
Prompt_VCR_QAR = [
    'Question: " {1} " Answer: " {2} " Rationale: '
]
Prompt_VisComet_Before = [
    'Event: " {1} {2} " Before, what the person needed to do ?'
]
Prompt_VisComet_Intent = [
    'Event: " {1} {2} " Because, what the person wanted to do ?'
]
Prompt_VisComet_After = [
    'Event: " {1} {2} " After, what the person will most likely to do ?'
]
Prompt_Ground_Situation_Recognition_Verb = [
    'What is the salient activity of the image ?'
]
Prompt_Ground_Situation_Recognition_Frame = [
    'Given the image and salient activity is " {1} ", what is the situation in terms of " {2} " ?'
]
Prompt_Segmentation_based_Image_Generation = [
    'What is the complete image? Segmentation color: " {} "'
]

#### TODO.md Template
This file tracks tasks, bugs, and features that need attention in your project.

# TODO List

This document tracks immediate and future tasks for the Facial Emotion Detection project.

## Immediate Tasks

### Critical

- [ ] Fix issue with image processing in MediaPipe integration
  - [ ] Handle "Please provide 'image_format' with 'data'" errors
  - [ ] Implement proper error handling for failed detections

- [ ] Improve model performance
  - [ ] Analyze confusion matrix for commonly misclassified emotions
  - [ ] Consider class weighting for imbalanced emotion categories

- [ ] Code optimization
  - [ ] Profile and optimize the landmark extraction process
  - [ ] Reduce memory usage during batch processing

### High Priority

- [ ] Add camera selection dropdown in test_model.py
  - [ ] Auto-detect available cameras
  - [ ] Save user preference

- [ ] Implement simple CLI arguments
  - [ ] Add --camera option to select camera index
  - [ ] Add --model option to specify model path

- [ ] Add basic logging
  - [ ] Create logging configuration
  - [ ] Add DEBUG level logs for development

## Medium Term

### Features

- [ ] Create visualization utilities
  - [ ] Generate graphs of emotion distribution
  - [ ] Show landmark points in 3D space

- [ ] Add alternative models
  - [ ] Support for SVM classifier
  - [ ] Support for CNN-based models
  - [ ] Model comparison utilities

- [ ] Implement emotion tracking over time
  - [ ] Track emotion transitions
  - [ ] Calculate emotion stability metrics
  - [ ] Generate time-series visualizations

### User Experience

- [ ] Improve visual feedback
  - [ ] Add confidence score display
  - [ ] Show emotion-specific visual cues
  - [ ] Implement emoji overlay option

- [ ] Enhance test_model.py UI
  - [ ] Add settings panel
  - [ ] Create recording capability
  - [ ] Implement screenshot functionality

## Long Term

### Major Features

- [ ] Multi-face processing
  - [ ] Track multiple people simultaneously
  - [ ] Assign unique IDs to each face
  - [ ] Provide comparative emotion analysis

- [ ] Web application
  - [ ] Create Flask/FastAPI backend
  - [ ] Develop simple frontend
  - [ ] Add real-time processing via WebRTC

- [ ] Video file processing
  - [ ] Add support for analyzing saved videos
  - [ ] Generate emotion summary reports
  - [ ] Create highlight markers for emotion changes

### Research

- [ ] Investigate emotion context awareness
  - [ ] Consider sequential information
  - [ ] Implement long-term tracking

- [ ] Explore cross-cultural emotion differences
  - [ ] Test on diverse datasets
  - [ ] Implement cultural adaptation

## Technical Debt

- [ ] Code cleanup
  - [ ] Standardize error handling
  - [ ] Improve comments and docstrings
  - [ ] Create unit tests

- [ ] Refactor utils.py
  - [ ] Split into logical modules
  - [ ] Improve MediaPipe integration
  - [ ] Create abstraction layer for landmark detection

- [ ] Documentation
  - [ ] Add function docstrings
  - [ ] Create API documentation
  - [ ] Improve README with examples
```

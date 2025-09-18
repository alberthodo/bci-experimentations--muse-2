# BCI Muse - Prototypes

A collection of brain compiuter interface protptypes done with the muse 2 headband.

- Detect raw artifacts
- ML based mental state tracker
- Mind controlled Orb with blinks and thoughts
- Music player controlled with thought



<br>

## Detecting raw artifacts
<br>

![8](https://github.com/user-attachments/assets/803653ec-117c-460e-ad4e-7ffd6b040bdb)

<br>
Prototyped a real-time EEG signal visualization system using the Muse 2 headset to explore raw neural data streams and identify viable input patterns. The implementation captures live EEG data across all available channels and renders them in a continuous scrolling display, allowing for immediate observation of various neural artifacts and noise patterns. 

During testing, I discovered that while most EEG signals contain significant artifacts and environmental noise, eye blinks consistently produced the cleanest and most distinguishable signal spikes across the channels. This observation led to the realization that blink detection could serve as a reliable, intentional input mechanism that operates alongside traditional EEG brainwave analysis.


The clarity and consistency of blink artifacts in the EEG stream opens up compelling possibilities for hybrid brain-computer interfaces that combine voluntary motor actions with passive neural monitoring. Similar to how Apple's Vision Pro leverages eye-tracking as a selection mechanism paired with hand gestures for confirmation, blink detection could provide a natural, low-latency input method that users can consciously control without the complexity of training machine learning models on subtle brainwave patterns. 

This  feels intuitive because blinking is already a natural, frequent action that can be easily modulated in timing and intensity, potentially enabling applications ranging from assistive technology interfaces to gaming controls where users maintain the familiarity of intentional physical input while benefiting from the seamless, contactless nature of neural sensing.

<br>

## Controlling an orb with the mind

<br>

![orb](https://github.com/user-attachments/assets/e965830f-0201-4b48-ac08-e6e4cb951cb6)

<br>

Inspired by the innitial control, I prototyped a Unity-based interface that transforms EEG mental states into real-time 3D visual feedback using the Muse headset. The system connects the Python BrainFlow server to Unity via UDP, enabling users to control a glowing orb through intentional double-blinks and passive mental state monitoring. Double-blinks toggle the orb's power state while focus and calm mental states dynamically adjust the orb's size, color, and rotation speed.

The experience demonstrates intuitive neural control by mapping natural mental states to compelling visual responses—focused attention makes the orb grow larger and spin faster, while a calm mind reduces its size and slows rotation. Double blink toggles on the orbs light. This creates an immediate feedback loop where users can observe their mental state changes in real-time, similar to biofeedback systems but with the engaging, immersive quality of interactive 3D graphics that feels both meditative and empowering.

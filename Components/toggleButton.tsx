import React, { useState } from 'react';
import { View, StyleSheet, TouchableOpacity, Animated, Easing } from 'react-native';

interface ToggleSwitchProps {
  onToggle: (value: boolean) => void; // Callback function to handle toggle state change
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({ onToggle }) => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const translateX = new Animated.Value(isConnected ? 30 : 0); // Initial position based on isConnected
  const circleColor = isConnected ? '#3498db' : '#f1c40f'; // Set circle color based on isConnected

  const toggleSwitch = () => {
    const newValue = !isConnected;
    setIsConnected(newValue);
    onToggle(newValue);

    // Animate the circle movement
    Animated.timing(translateX, {
      toValue: newValue ? 30 : 0,
      duration: 300,
      easing: Easing.linear,
      useNativeDriver: true,
    }).start();
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.switchButton} onPress={toggleSwitch}>
        {/* Circle indicator */}
        <Animated.View style={[styles.circle, { transform: [{ translateX }], backgroundColor: circleColor }]} />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  switchButton: {
    width: 49,
    height: 26,
    borderRadius: 15,
    backgroundColor: '#CCCCCC',
    justifyContent: 'center',
  },
  circle: {
    width: 29,
    height: 26,
    borderRadius: 13,
  },
});

export default ToggleSwitch;

// TimerControls.tsx
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

interface TimerControlsProps {
  onStart: () => void;
  onPause: () => void;
  onRefresh: () => void;
}

const TimerControls: React.FC<TimerControlsProps> = ({ onStart, onPause, onRefresh }) => {
  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={onStart} style={styles.button}>
        <Text>Play</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={onPause} style={styles.button}>
        <Text>Pause</Text>
      </TouchableOpacity>
      <TouchableOpacity onPress={onRefresh} style={styles.button}>
        <Text>Refresh</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    marginVertical: 10,
  },
  button: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    backgroundColor: '#ccc',
    borderRadius: 5,
    marginHorizontal: 5,
  },
});

export default TimerControls;

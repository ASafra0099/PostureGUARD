import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';

const SetgoalScreen: React.FC = () => {
  const [hours, setHours] = useState<number>(0);
  const [minutes, setMinutes] = useState<number>(0);
  const [seconds, setSeconds] = useState<number>(0);

  // Function to start the timer
  const startTimer = () => {
    // Start timer logic here
  };

  return (
    <View style={styles.container}>
      <View style={styles.timeContainer}>
        <Text style={styles.title}>Set Your Goal</Text>
        <View style={styles.timeInput}>
          <ScrollView>
            <Text style={styles.timeLabel}>Hours</Text>
            <Text style={styles.timeValue}>{hours}</Text>
            {/* Add logic to set hours */}
          </ScrollView>
          <ScrollView>
            <Text style={styles.timeLabel}>Minutes</Text>
            <Text style={styles.timeValue}>{minutes}</Text>
            {/* Add logic to set minutes */}
          </ScrollView>
          <ScrollView>
            <Text style={styles.timeLabel}>Seconds</Text>
            <Text style={styles.timeValue}>{seconds}</Text>
            {/* Add logic to set seconds */}
          </ScrollView>
        </View>
      </View>
      <TouchableOpacity style={styles.startButton} onPress={startTimer}>
        <Text style={styles.startButtonText}>Start</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
  },
  timeContainer: {
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  timeInput: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '80%',
  },
  timeLabel: {
    fontSize: 18,
    marginBottom: 5,
  },
  timeValue: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  startButton: {
    marginTop: 40,
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#47abea',
    borderRadius: 8,
  },
  startButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
});

export default SetgoalScreen;

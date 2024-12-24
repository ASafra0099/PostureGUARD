// ScrollableSelection.tsx
import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';

interface ScrollableSelectionProps {
  items: number[];
  selectedValue: number | null;
  onStartTimer: (value: number) => void;
}

const ScrollableSelection: React.FC<ScrollableSelectionProps> = ({ items, selectedValue, onStartTimer }) => {
  const itemHeight = 50; // Adjust this value according to your item height
  const timerContentHeight = items.length * itemHeight;
  const boxHeight = itemHeight * 3; // Height of the single box covering hours, minutes, and seconds

  return (
    <View style={styles.container}>
      <View style={[styles.activeBox, { top: timerContentHeight / 2 - boxHeight / 2 }]} />
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={[styles.scrollViewContent, { height: timerContentHeight }]}>
        {items.map((item, index) => (
          <TouchableOpacity
            key={index}
            onPress={() => onStartTimer(item)}
            style={selectedValue === item ? styles.selectedItem : null}>
            <Text style={styles.item}>{item}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: '30%',
    height: '50%',
    borderWidth: 1, // Add border for visualization
    position: 'relative',
  },
  scrollViewContent: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  item: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    color: 'black',
    textAlign: 'center',
  },
  selectedItem: {
    backgroundColor: 'lightblue', // Add background color for selected item
  },
  activeBox: {
    position: 'absolute',
    backgroundColor: 'lightblue', // Add background color for active box
    width: '100%',
    height: 150, // Adjust height as needed
  },
});

export default ScrollableSelection;

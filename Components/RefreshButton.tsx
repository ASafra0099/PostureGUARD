import React from 'react';
import { TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface RefreshButtonProps {
  onPress: () => void;
}

const RefreshButton: React.FC<RefreshButtonProps> = ({ onPress }) => {
  return (
    <TouchableOpacity onPress={onPress}>
      <Ionicons name="sync" size={30} color="#47abea" />
    </TouchableOpacity>
  );
};

export default RefreshButton;

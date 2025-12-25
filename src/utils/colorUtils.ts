export function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  // Convert to HSL for better control over vibrancy
  // Use a fixed saturation and lightness to ensure all colors look good
  const hue = Math.abs(hash % 360);
  const saturation = 70; // 70% saturation for vibrancy
  const lightness = 60;  // 60% lightness for visibility on both dark/light modes
  
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

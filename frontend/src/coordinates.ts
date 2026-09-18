import type { Box, Element, Point, Screenshot } from './types';
export const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));
export const center = (b: Box): Point => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
export const orderBox = (a: Point, b: Point): Box => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
export const normalizedToPixels = (point: Point, width: number, height: number): Point => [point[0] * width, point[1] * height];
export const pixelsToNormalized = (point: Point, width: number, height: number): Point => [point[0] / width, point[1] / height];
export const contains = (b: Box, p: Point) => p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
export function moveElement(element: Element, dx: number, dy: number): Element {
  const x = clamp(dx, -element.bbox[0], 1 - element.bbox[2]);
  const y = clamp(dy, -element.bbox[1], 1 - element.bbox[3]);
  return { ...element, bbox: [element.bbox[0] + x, element.bbox[1] + y, element.bbox[2] + x, element.bbox[3] + y], click_point: [element.click_point[0] + x, element.click_point[1] + y] };
}
export function resizeElement(element: Element, corner: number, point: Point): Element {
  const opposite: Point = [element.bbox[corner === 0 || corner === 3 ? 2 : 0], element.bbox[corner < 2 ? 3 : 1]];
  const bbox = orderBox(opposite, [clamp(point[0]), clamp(point[1])]);
  return { ...element, bbox, click_point: [clamp(element.click_point[0], bbox[0], bbox[2]), clamp(element.click_point[1], bbox[1], bbox[3])] };
}
export function annotationErrors(image: Screenshot, classes: string[]): string[] {
  const errors: string[] = [];
  for (const [i, element] of image.elements.entries()) {
    const b = element.bbox;
    if (b.some(v => !Number.isFinite(v) || v < 0 || v > 1) || b[2] <= b[0] || b[3] <= b[1]) errors.push(`Element ${i + 1}: box must have positive width and height and stay inside the image.`);
    if (!classes[element.class_id]) errors.push(`Element ${i + 1}: choose an available class.`);
    if (!element.click_point.every(Number.isFinite) || !contains(b, element.click_point)) errors.push(`Element ${i + 1}: click point must be inside its target box.`);
  }
  for (const [i, example] of image.examples.entries()) {
    if (!example.instruction.trim()) errors.push(`Instruction ${i + 1}: enter an instruction.`);
    if (example.target_present && !image.elements.some(e => e.id === example.element_id)) errors.push(`Instruction ${i + 1}: choose a target element or mark it absent.`);
    if (example.ambiguous && example.status === 'reviewed') errors.push(`Instruction ${i + 1}: ambiguous examples must remain draft or excluded.`);
  }
  return errors;
}

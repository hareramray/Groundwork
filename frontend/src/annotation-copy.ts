import type { Element, Example, Screenshot } from './types';

export interface CopyOptions {
  elementIds?: string[];
  includeAbsent?: boolean;
}

/** Copy saved annotations into an independent draft, preserving normalized geometry. */
export function copyAnnotations(
  source: Screenshot,
  options: CopyOptions = {},
  createId: () => string = () => crypto.randomUUID(),
): { elements: Element[]; examples: Example[] } {
  const available = new Set(source.elements.map(element => element.id));
  const selected = new Set(options.elementIds ?? available);
  if ([...selected].some(id => !available.has(id))) {
    throw new Error('A selected element is no longer available. Choose the source image again.');
  }
  if (available.size !== source.elements.length || source.examples.some(example =>
    example.target_present && (!example.element_id || !available.has(example.element_id)))) {
    throw new Error('The source image has invalid instruction targets. Correct its annotations before copying.');
  }

  const replacements = new Map<string, string>();
  const elements = source.elements.filter(element => selected.has(element.id)).map(element => {
    const id = createId();
    replacements.set(element.id, id);
    // Coordinates are normalized, so the same box scales with the destination image.
    return { ...structuredClone(element), id };
  });
  const examples: Example[] = source.examples.filter(example => example.target_present
    ? replacements.has(example.element_id!)
    : (options.includeAbsent ?? true)).map(example => ({
    ...structuredClone(example),
    id: createId(),
    element_id: example.target_present ? replacements.get(example.element_id!)! : null,
    status: 'draft',
  }));

  return { elements, examples };
}

import { describe, expect, it } from 'vitest';
import { annotationErrors, contains, moveElement, normalizedToPixels, pixelsToNormalized, resizeElement } from './coordinates';
import type { Element, Screenshot } from './types';
const element: Element = { id: 'e1', class_id: 0, label: 'Submit', bbox: [.2, .3, .6, .7], click_point: [.4, .5] };
describe('annotation coordinate invariants', () => {
  it('round trips original pixels independently of display size and zoom', () => {
    for (const [width, height] of [[1920, 1080], [320, 200], [4000, 800]]) {
      const normalized = pixelsToNormalized([width * .3125, height * .75], width, height);
      expect(normalizedToPixels(normalized, width, height)).toEqual([width * .3125, height * .75]);
      expect(normalized).toEqual([.3125, .75]);
    }
  });
  it('clamps moves while preserving dimensions and click-point offset', () => {
    const moved = moveElement(element, 2, -2);
    expect(moved.bbox[2]).toBe(1); expect(moved.bbox[1]).toBe(0);
    expect(moved.bbox[2] - moved.bbox[0]).toBeCloseTo(.4);
    expect(moved.click_point[0] - moved.bbox[0]).toBeCloseTo(.2);
    expect(contains(moved.bbox, moved.click_point)).toBe(true);
  });
  it('allows corner crossings and keeps click point inside resized box', () => {
    const resized = resizeElement(element, 0, [.8, .9]);
    expect(resized.bbox).toEqual([.6, .7, .8, .9]);
    expect(contains(resized.bbox, resized.click_point)).toBe(true);
  });
  it('rejects invalid targets, unknown classes, empty instructions and external click points', () => {
    const image = { elements: [{ ...element, class_id: 99, click_point: [1, 1] }], examples: [{ instruction: '', target_present: true, element_id: 'missing', status: 'draft' }] } as Screenshot;
    expect(annotationErrors(image, ['button'])).toHaveLength(4);
  });
  it('accepts a reviewed absent example with no element', () => {
    const image = { elements: [], examples: [{ instruction: 'Find settings', target_present: false, element_id: null, status: 'reviewed', ambiguous: false }] } as Screenshot;
    expect(annotationErrors(image, ['button'])).toEqual([]);
  });
});

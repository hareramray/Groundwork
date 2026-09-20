import { describe, expect, it } from 'vitest';
import { copyAnnotations } from './annotation-copy';
import { annotationErrors, contains, normalizedToPixels, resizeElement } from './coordinates';
import type { Screenshot } from './types';

const source: Screenshot = {
  id: 'source', filename: 'desktop.png', width: 1600, height: 900, url: '/source.png',
  group: 'same-page', status: 'draft', synthetic: false,
  elements: [
    { id: 'button', class_id: 0, label: 'Submit', bbox: [.2, .3, .6, .7], click_point: [.5, .6] },
    { id: 'field', class_id: 1, label: 'Search', bbox: [.1, .1, .5, .2], click_point: [.2, .15] },
  ],
  examples: [
    { id: 'submit-1', instruction: 'Click Submit', element_id: 'button', target_present: true, status: 'reviewed', ambiguous: false },
    { id: 'submit-2', instruction: 'Send the form', element_id: 'button', target_present: true, status: 'excluded', ambiguous: false },
    { id: 'search', instruction: 'Find search', element_id: 'field', target_present: true, status: 'draft', ambiguous: true },
    { id: 'absent', instruction: 'Find the account menu', element_id: null, target_present: false, status: 'reviewed', ambiguous: false },
  ],
};

function ids() { let value = 0; return () => `copy-${++value}`; }

describe('copying annotations between screenshots', () => {
  it('creates independent IDs and preserves multiple instructions for the same copied target', () => {
    const before = structuredClone(source);
    const result = copyAnnotations(source, {}, ids());
    expect(result.elements.map(element => element.id)).toEqual(['copy-1', 'copy-2']);
    expect(result.examples.map(example => example.id)).toEqual(['copy-3', 'copy-4', 'copy-5', 'copy-6']);
    expect(result.examples.map(example => example.element_id)).toEqual(['copy-1', 'copy-1', 'copy-2', null]);
    expect(result.examples.every(example => example.status === 'draft')).toBe(true);
    expect(result.examples[2].ambiguous).toBe(true);
    expect(result.elements[0]).toEqual({ ...source.elements[0], id: 'copy-1' });
    result.elements[0].bbox[0] = .05;
    result.elements[0].click_point[0] = .1;
    result.examples[0].instruction = 'Changed for a new page';
    expect(source).toEqual(before);
  });

  it('copies only selected elements and their linked instructions, with optional absent examples', () => {
    const selected = copyAnnotations(source, { elementIds: ['button'], includeAbsent: false }, ids());
    expect(selected.elements).toHaveLength(1);
    expect(selected.examples.map(example => example.instruction)).toEqual(['Click Submit', 'Send the form']);
    expect(selected.examples.every(example => example.element_id === selected.elements[0].id)).toBe(true);
    const withAbsent = copyAnnotations(source, { elementIds: ['button'] }, ids());
    expect(withAbsent.examples).toHaveLength(3);
    expect(withAbsent.examples.at(-1)?.target_present).toBe(false);
    expect(copyAnnotations(source, { elementIds: [], includeAbsent: false }, ids())).toEqual({ elements: [], examples: [] });
  });

  it('can copy absent-target instructions from an image with no boxes', () => {
    const result = copyAnnotations({ ...source, elements: [], examples: [source.examples[3]] }, {}, ids());
    expect(result.elements).toEqual([]);
    expect(result.examples[0]).toEqual({ ...source.examples[3], id: 'copy-1', status: 'draft' });
  });

  it('scales to a different aspect ratio and allows resizing while retaining valid target links', () => {
    const result = copyAnnotations(source, { elementIds: ['button'], includeAbsent: false }, ids());
    const destination: Screenshot = { ...source, id: 'destination', width: 400, height: 800, ...result };
    expect(normalizedToPixels([result.elements[0].bbox[2], result.elements[0].bbox[3]], 400, 800)).toEqual([240, 560]);
    destination.elements[0] = resizeElement(destination.elements[0], 2, [.35, .45]);
    expect(destination.elements[0].bbox).toEqual([.2, .3, .35, .45]);
    expect(contains(destination.elements[0].bbox, destination.elements[0].click_point)).toBe(true);
    expect(annotationErrors(destination, ['button', 'textbox'])).toEqual([]);
    expect(source.elements[0].bbox).toEqual([.2, .3, .6, .7]);
  });

  it('rejects stale selections and invalid source links instead of copying dangling instructions', () => {
    expect(() => copyAnnotations(source, { elementIds: ['deleted'] })).toThrow('no longer available');
    expect(() => copyAnnotations({ ...source, elements: [] })).toThrow('invalid instruction targets');
  });

  it('repeated copies do not share element or instruction IDs', () => {
    const createId = ids();
    const one = copyAnnotations(source, {}, createId);
    const two = copyAnnotations(source, {}, createId);
    const allIds = [...one.elements, ...one.examples, ...two.elements, ...two.examples].map(item => item.id);
    expect(new Set(allIds).size).toBe(allIds.length);
  });
});

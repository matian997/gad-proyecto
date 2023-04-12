CREATE OR REPLACE FUNCTION DISTANCE(L real[], R real[]) RETURNS real AS $$
DECLARE
  s real;
BEGIN
  s := 0;
  FOR i IN 0..array_length(l, 1) LOOP
    s := s + power((l[i] - r[i]), 2);
  END LOOP;
  RETURN sqrt(s);
END;

$$ LANGUAGE PLPGSQL;


CREATE TABLE PIVOTES(
	ID SERIAL,
	IMAGEID SERIAL,
	PRIMARY KEY (ID),
	FOREIGN KEY (IMAGEID) REFERENCES IMAGES(ID));

CREATE TABLE NODES(
	ID SERIAL, 
	PARENTID SMALLINT NULL,
	DISTANCE real, 
	PIVOTEID SMALLINT NULL,
	PRIMARY KEY (ID),
	FOREIGN KEY (PARENTID) REFERENCES NODES(ID),
	FOREIGN KEY (PIVOTEID) REFERENCES PIVOTES(ID));

CREATE TABLE IMAGES_DATASET(
	ID SERIAL, 
	NAME VARCHAR, 
	HISTOGRAM real[], 
	PRIMARY KEY (ID));
	
CREATE TABLE IMAGES(
	ID SERIAL, 
	IMAGE_ID INTEGER,
	NODE_ID INTEGER, 
	PRIMARY KEY (ID),
	FOREIGN KEY (IMAGE_ID) REFERENCES IMAGES_DATASET(ID)
	FOREIGN KEY (NODE_ID) REFERENCES NODES(ID));

CREATE OR REPLACE FUNCTION DELETE_IMAGE() RETURNS TRIGGER AS $body$
DECLARE
	nodosMismoNivel int;
	nodoHijos int;
	idNodoPadre int;
	nextIdNodoPadre int;
BEGIN
	nodoHijos = (
		SELECT COUNT(*)
		FROM nodes
		WHERE parentId = OLD.nodeId);

	IF (nodoHijos = 0) THEN

	idNodoPadre := (
		SELECT parentId
		FROM nodes
		WHERE nodeId = OLD.nodeId
		LIMIT 1);

		DELETE FROM nodes WHERE (nodeId = OLD.nodeId);

	nodosMismoNivel := (
			SELECT COUNT(*)
			FROM nodes
			WHERE parentId = idNodoPadre);

	WHILE nodosMismoNivel = 0 LOOP
			nextIdNodoPadre := (
				SELECT parentId
				FROM nodes
				WHERE nodeId = idNodoPadre
				LIMIT 1);

			IF (nextIdNodoPadre IS not null) THEN

			DELETE FROM nodes WHERE (nodeId = idNodoPadre);
			idNodoPadre := nextIdNodoPadre;
			END IF;

			nodosMismoNivel := (
			SELECT COUNT(*)
			FROM nodes
			WHERE parentId = idNodoPadre);

	END LOOP;
	ELSE
		RETURN NULL;
	END IF;

	RETURN OLD;
END
$body$ LANGUAGE 'plpgsql';


CREATE TRIGGER DELETE_IMAGE AFTER
DELETE ON IMAGES
FOR EACH ROW EXECUTE PROCEDURE DELETE_IMAGE();


CREATE OR REPLACE FUNCTION ADD_IMAGE() RETURNS TRIGGER AS $body$
DECLARE
	distanciax SMALLINT;
	pivote_record RECORD;
	idNodoRaiz SMALLINT;
	idNodox SMALLINT;
	pivote_image: images
BEGIN
	idNodoRaiz := 1;
	FOR pivote_record IN SELECT * FROM pivotes
		LOOP
			pivote_image := (
				SELECT HISTOGRAM
				FROM images
				WHERE id = pivote_record.imaggeId
				LIMIT 1);

			distanciax := DISTANCE(pivote_image.HISTOGRAM, NEW.HISTOGRAM);

			idNodox := (
				SELECT nodeId
				FROM nodes
				WHERE parentId = idNodoRaiz AND distance = distanciax
				LIMIT 1);

			IF (idNodox IS NULL) THEN
				INSERT INTO nodes(parentId, distance, pivoteId) VALUES (
					idNodoRaiz,
					distanciax,
					pivote_record.pivoteId);

				idNodoRaiz := (
					SELECT nodeId
					FROM nodes
					WHERE parentId = idNodoRaiz AND distance = distanciax
					LIMIT 1);
			ELSE
				idNodoRaiz := idNodox;
			END IF;
		END LOOP;

	NEW.nodeId := idNodoRaiz;

	RETURN NEW;
END
$body$ LANGUAGE 'plpgsql';


CREATE TRIGGER ADD_IMAGE
BEFORE
INSERT ON IMAGES
FOR EACH ROW EXECUTE PROCEDURE ADD_IMAGE();


CREATE OR REPLACE FUNCTION EDIT_IMAGE() RETURNS TRIGGER AS $body$
BEGIN
	DELETE FROM images WHERE (name = OLD.name);
	INSERT INTO images(name) values (NEW.name);

	RETURN NEW;
END
$body$ LANGUAGE 'plpgsql';

CREATE TRIGGER EDIT_IMAGE AFTER
UPDATE ON NOMBRES
FOR EACH ROW EXECUTE PROCEDURE EDIT_IMAGE();


-- "incremental_pivot_selection"
DECLARE
	pivots int[];
	sample int[];
	pairs int[];
	candidate int;
	average float;
	max_average float;
	winner int;
BEGIN
	pivots := array[]::int[];
	WHILE (cardinality(pivots) < n_pivots) LOOP
		sample := ARRAY(SELECT id FROM images WHERE ID != ALL(pivots) ORDER BY random() LIMIT k_sample);
		pairs := ARRAY(SELECT id FROM images  WHERE ID != ALL(pivots) AND ID != ALL(sample) ORDER BY random() LIMIT a_pairs*2);
		max_average := 0.0;
		winner := sample[0];
		FOREACH candidate IN ARRAY sample LOOP
			average := candidate_mu(pivots, candidate, pairs);
			IF average > max_average THEN
				max_average := average;
				winner := candidate;
			END IF;
		END LOOP;
	pivots := array_append(pivots, winner);
	END LOOP;
	RETURN pivots;
END; 


-- "candidate_mu"
DECLARE
	pivotId int;
	pivot int[];
	i int;
	Aelement int[];
	Belement int[];
	max_diff float;
	diff float;
	total float;
BEGIN
	pivots := array_append(pivots, candidate);
	total := 0.0;
	i := 1;
	WHILE (i < cardinality(pairs)) LOOP
		Aelement := (SELECT HISTOGRAM FROM IMAGES WHERE ID = pairs[i] LIMIT 1);
		Belement := (SELECT HISTOGRAM FROM IMAGES WHERE ID = pairs[i+1] LIMIT 1);
		max_diff := 0.0;
		RAISE NOTICE 'Pivots: %', cardinality(pivots);
		RAISE NOTICE 'pairs: %', cardinality(pairs);
		RAISE NOTICE 'I: %', i;
		RAISE NOTICE 'pairs[i]: %', pairs[i];
		FOREACH pivotId IN ARRAY pivots LOOP
			RAISE NOTICE 'Pivot: %', pivotId;
			pivot := (SELECT HISTOGRAM FROM IMAGES WHERE ID = pivotId LIMIT 1);
			RAISE NOTICE 'Pivot h: %', pivot;
			RAISE NOTICE 'Aelement: %', Aelement;
			RAISE NOTICE 'Belement: %', Belement;
			diff := abs(distance(Aelement, pivot) - distance(Belement, pivot))::float;
			RAISE NOTICE 'diff: %', diff;
			IF diff > max_diff THEN
				max_diff := diff;
			END IF;
		END LOOP;
		total := total + max_diff;
		i := i + 2;
	END LOOP;
	RETURN total/(cardinality(pairs)::float/2.0);
END; 


CREATE OR REPLACE FUNCTION CREATE_PIVOTS(n_pivots integer, k_samples integer, a_pairs integer) 
RETURNS void AS $$
DECLARE
  pivot_ids integer[];
  pivot_id integer;
  vector integer[];
BEGIN
	pivot_ids := Array(select incremental_pivot_selection(n_pivots, k_samples, a_pairs));
	FOREACH pivot_id IN ARRAY pivot_ids LOOP
		vector := (SELECT HISTOGRAM FROM IMAGES WHERE ID = pivot_id LIMIT 1);
		INSERT INTO PIVOTES(HISTOGRAM) VALUES (vector);
	END LOOP;
END;
$$ LANGUAGE PLPGSQL;
-- Permite regenerar zona_clusters sin intentar dejar ciudad en NULL.
-- Necesario solo para bases creadas antes de esta corrección.

BEGIN;

ALTER TABLE property_analytics
    DROP CONSTRAINT IF EXISTS fk_cluster_ciudad;

ALTER TABLE property_analytics
    ADD CONSTRAINT fk_cluster_ciudad
    FOREIGN KEY (cluster_zona, ciudad)
    REFERENCES zona_clusters(cluster_id, ciudad)
    ON DELETE SET NULL (cluster_zona)
    DEFERRABLE INITIALLY DEFERRED;

COMMIT;
